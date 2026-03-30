"""
probe_internals.py: Extract hidden-state activations and train linear probes.

This module implements the Internal-Surface Divergence (ISD) component:
  1. Extract activations at selected layers for each response
  2. Train linear probes to predict correctness from activations
  3. Compute ISD: probe confidence in "correct" for actually-incorrect responses
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """Per-layer probe performance."""
    layer_idx: int
    accuracy: float
    auc: float
    n_train: int
    n_test: int


@dataclass
class ISDScore:
    """ISD score for a single response."""
    task_id: str
    domain: str
    is_correct: bool
    layer_scores: dict[int, float]   # layer_idx → probe P(correct)
    isd_score: float                  # aggregated ISD (mean across layers)


# ---------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------

def select_layers(n_layers: int) -> list[int]:
    """Select 5 evenly-spaced layers including first and last."""
    indices = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    return sorted(set(indices))


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_activations(
    model,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    layer_indices: list[int],
    batch_size: int = 4,
    position: str = "last_token",
    format_fn=None,
) -> dict[int, np.ndarray]:
    """
    Extract hidden-state activations at the final token of each response.

    Args:
        model: the HuggingFace model (same one used for generation).
        tokenizer: corresponding tokenizer.
        prompts: raw user prompts (before chat-template formatting).
        responses: model-generated response strings.
        layer_indices: which layers to extract from.
        batch_size: inference batch size (keep small; forward pass
            with output_hidden_states=True uses ~2x the memory of
            generation).
        position: "last_token" to probe at the final non-padding
            position (recommended).
        format_fn: callable(tokenizer, prompt) → formatted string.
            If provided, each prompt is formatted through this before
            concatenation with the response.  **This must match the
            formatting used during generation**; otherwise the token
            positions diverge and the activations are meaningless.

    Returns:
        dict mapping layer_idx → array of shape (n_samples, hidden_dim).
    """
    activations = {layer: [] for layer in layer_indices}

    for start in tqdm(range(0, len(prompts), batch_size), desc="Extracting activations"):
        batch_prompts = prompts[start : start + batch_size]
        batch_responses = responses[start : start + batch_size]

        # Reconstruct the exact token sequence the model saw during generation
        if format_fn is not None:
            full_texts = [format_fn(tokenizer, p) + r for p, r in zip(batch_prompts, batch_responses)]
        else:
            full_texts = [p + r for p, r in zip(batch_prompts, batch_responses)]

        inputs = tokenizer(
            full_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=1280,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden)

        for j in range(len(batch_prompts)):
            # Find the last non-padding token position
            attention_mask = inputs["attention_mask"][j]
            if position == "last_token":
                last_pos = attention_mask.sum().item() - 1
            else:
                last_pos = -1

            for layer_idx in layer_indices:
                act = hidden_states[layer_idx][j, last_pos, :].cpu().float().numpy()
                activations[layer_idx].append(act)

        # Free GPU memory between batches
        del outputs, hidden_states, inputs
        torch.cuda.empty_cache()

    # Stack into arrays
    return {k: np.stack(v) for k, v in activations.items()}


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def train_probes(
    activations: dict[int, np.ndarray],
    labels: np.ndarray,
    test_split: float = 0.2,
    random_seed: int = 42,
) -> tuple[dict[int, tuple[LogisticRegression, StandardScaler]], list[ProbeResult]]:
    """
    Train a logistic regression probe per layer.

    Activations are standardised (zero mean, unit variance) before
    fitting.  Each layer gets its own scaler, fitted on the training
    split only.  The returned dict maps layer_idx → (probe, scaler)
    so that the same transform can be applied at ISD scoring time.

    Args:
        activations: layer_idx → (n_samples, hidden_dim) array
        labels: binary array (1 = correct, 0 = incorrect)
        test_split: fraction for held-out evaluation
        random_seed: for reproducibility

    Returns:
        probes: layer_idx → (fitted LogisticRegression, fitted StandardScaler)
        results: per-layer performance metrics
    """
    probes = {}
    results = []

    X_train_idx, X_test_idx = train_test_split(
        np.arange(len(labels)), test_size=test_split,
        random_state=random_seed, stratify=labels,
    )

    for layer_idx, X in activations.items():
        X_train, X_test = X[X_train_idx], X[X_test_idx]
        y_train, y_test = labels[X_train_idx], labels[X_test_idx]

        # Standardise: fit on train only, apply to both
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        probe = LogisticRegression(
            max_iter=2000, solver="lbfgs", C=1.0, random_state=random_seed,
        )
        probe.fit(X_train, y_train)

        y_pred = probe.predict(X_test)
        y_prob = probe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5  # edge case: single class in test set

        probes[layer_idx] = (probe, scaler)
        results.append(ProbeResult(
            layer_idx=layer_idx, accuracy=acc, auc=auc,
            n_train=len(X_train), n_test=len(X_test),
        ))
        logger.info(f"Layer {layer_idx}: accuracy={acc:.3f}, AUC={auc:.3f}")

    return probes, results


# ---------------------------------------------------------------------------
# ISD scoring
# ---------------------------------------------------------------------------

def compute_isd(
    activations: dict[int, np.ndarray],
    probes: dict[int, tuple],
    task_ids: list[str],
    domains: list[str],
    labels: np.ndarray,
) -> list[ISDScore]:
    """
    Compute ISD scores for all responses.

    ISD = probe's predicted P(correct) for each response.
    For errors, high ISD means the model internally "knows" the right answer.

    probes: layer_idx → (LogisticRegression, StandardScaler)
    """
    scores = []

    for i in range(len(task_ids)):
        layer_scores = {}
        for layer_idx, (probe, scaler) in probes.items():
            x = activations[layer_idx][i].reshape(1, -1)
            x = scaler.transform(x)
            p_correct = probe.predict_proba(x)[0, 1]
            layer_scores[layer_idx] = float(p_correct)

        # Aggregate: mean probe confidence across layers
        isd = float(np.mean(list(layer_scores.values())))

        scores.append(ISDScore(
            task_id=task_ids[i],
            domain=domains[i],
            is_correct=bool(labels[i]),
            layer_scores=layer_scores,
            isd_score=isd,
        ))

    return scores


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_activations(activations: dict[int, np.ndarray], path: str | Path) -> None:
    """Save activations as compressed numpy archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{f"layer_{k}": v for k, v in activations.items()},
    )
    logger.info(f"Saved activations to {path}")


def load_activations(path: str | Path) -> dict[int, np.ndarray]:
    """Load activations from numpy archive."""
    data = np.load(path)
    return {int(k.split("_")[1]): data[k] for k in data.files}


def save_probes(probes: dict[int, LogisticRegression], path: str | Path) -> None:
    """Save trained probes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(probes, f)
    logger.info(f"Saved probes to {path}")


def load_probes(path: str | Path) -> dict[int, LogisticRegression]:
    """Load trained probes."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_isd_scores(scores: list[ISDScore], path: str | Path) -> None:
    """Save ISD scores to JSON Lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in scores:
            f.write(json.dumps(asdict(s)) + "\n")
    logger.info(f"Saved {len(scores)} ISD scores to {path}")
