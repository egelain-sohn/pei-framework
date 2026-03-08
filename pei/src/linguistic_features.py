"""
linguistic_features.py — Extract linguistic confidence features from model responses.

Implements the Linguistic Confidence Score (LCS) component using a
principled taxonomy of surface-level features that capture how confidently
the model presents its answer.

Feature categories:
  1. Epistemic stance (boosters vs hedges)
  2. Evidentiality markers (direct claims vs hedged attribution)
  3. Discourse-level confidence (structure, definiteness)
  4. Syntactic assertiveness (declarative framing, voice)
  5. Fluency (sentence structure, repetition)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import spacy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lexicons — linguistically principled, not naive keyword lists
# ---------------------------------------------------------------------------

# Epistemic boosters: markers that strengthen assertion
BOOSTERS = {
    "certainly", "definitely", "clearly", "undoubtedly", "obviously",
    "without question", "without doubt", "unquestionably", "absolutely",
    "indeed", "of course", "surely", "no doubt", "in fact",
    "it is clear that", "it is evident that", "there is no question",
    "precisely", "exactly", "always", "never",
}

# Epistemic hedges: markers that weaken assertion
HEDGES = {
    "perhaps", "maybe", "might", "possibly", "arguably", "potentially",
    "it seems", "it appears", "I think", "I believe", "I suppose",
    "roughly", "approximately", "somewhat", "to some extent",
    "in some cases", "it could be", "likely", "unlikely",
    "not entirely clear", "uncertain", "debatable",
    "could", "would", "may",
}

# Direct evidence claims
DIRECT_EVIDENCE = {
    "studies show", "research demonstrates", "evidence indicates",
    "data shows", "according to research", "has been shown",
    "is well established", "is well documented", "proves that",
    "demonstrates that", "confirms that",
}

# Indirect / hedged evidence
INDIRECT_EVIDENCE = {
    "reportedly", "according to some", "it is said",
    "some suggest", "it has been suggested", "anecdotally",
    "some believe", "there is some evidence", "preliminary findings",
}


# ---------------------------------------------------------------------------
# Feature extraction functions
# ---------------------------------------------------------------------------

@dataclass
class LinguisticFeatures:
    """Full feature vector for a single response."""
    task_id: str
    domain: str

    # Epistemic stance
    n_boosters: int = 0
    n_hedges: int = 0
    booster_ratio: float = 0.0  # boosters / (boosters + hedges)

    # Evidentiality
    n_direct_evidence: int = 0
    n_indirect_evidence: int = 0
    n_bald_assertions: int = 0  # sentences with no evidential marking

    # Discourse confidence
    claim_in_first_sentence: bool = False
    n_concessive_structures: int = 0
    definite_reference_ratio: float = 0.0

    # Syntactic assertiveness
    declarative_ratio: float = 0.0
    active_voice_ratio: float = 0.0

    # Fluency
    mean_sentence_length: float = 0.0
    n_sentences: int = 0
    has_repetition: bool = False

    # Composite
    lcs_score: float = 0.0


def count_markers(text: str, marker_set: set[str]) -> int:
    """Count occurrences of markers in text (case-insensitive)."""
    text_lower = text.lower()
    count = 0
    for marker in marker_set:
        # Use word boundaries for single words, substring for phrases
        if " " in marker:
            count += text_lower.count(marker)
        else:
            count += len(re.findall(rf"\b{re.escape(marker)}\b", text_lower))
    return count


def extract_epistemic_features(text: str) -> dict:
    """Extract epistemic stance features."""
    n_boosters = count_markers(text, BOOSTERS)
    n_hedges = count_markers(text, HEDGES)
    total = n_boosters + n_hedges
    booster_ratio = n_boosters / total if total > 0 else 0.5  # neutral default

    return {
        "n_boosters": n_boosters,
        "n_hedges": n_hedges,
        "booster_ratio": booster_ratio,
    }


def extract_evidentiality_features(text: str, doc) -> dict:
    """Extract evidentiality markers."""
    n_direct = count_markers(text, DIRECT_EVIDENCE)
    n_indirect = count_markers(text, INDIRECT_EVIDENCE)

    # Count bald assertions: sentences with no evidential marking
    sentences = list(doc.sents)
    n_bald = 0
    for sent in sentences:
        sent_text = sent.text
        has_direct = count_markers(sent_text, DIRECT_EVIDENCE) > 0
        has_indirect = count_markers(sent_text, INDIRECT_EVIDENCE) > 0
        if not has_direct and not has_indirect:
            n_bald += 1

    return {
        "n_direct_evidence": n_direct,
        "n_indirect_evidence": n_indirect,
        "n_bald_assertions": n_bald,
    }


def extract_discourse_features(text: str, doc) -> dict:
    """Extract discourse-level confidence features."""
    sentences = list(doc.sents)

    # Check if the main claim is in the first sentence
    # (heuristic: first sentence contains no hedge markers)
    claim_first = False
    if sentences:
        first_sent = sentences[0].text
        claim_first = count_markers(first_sent, HEDGES) == 0

    # Concessive structures
    concessives = {"although", "however", "on the other hand", "nevertheless",
                   "nonetheless", "despite", "in spite of", "while it is true",
                   "admittedly", "granted"}
    n_concessive = count_markers(text, concessives)

    # Definite vs indefinite reference
    n_definite = len(re.findall(r"\bthe\b", text.lower()))
    n_indefinite = len(re.findall(r"\b(a|an)\b", text.lower()))
    total_ref = n_definite + n_indefinite
    definite_ratio = n_definite / total_ref if total_ref > 0 else 0.5

    return {
        "claim_in_first_sentence": claim_first,
        "n_concessive_structures": n_concessive,
        "definite_reference_ratio": definite_ratio,
    }


def extract_syntactic_features(doc) -> dict:
    """Extract syntactic assertiveness features."""
    sentences = list(doc.sents)
    if not sentences:
        return {"declarative_ratio": 0.5, "active_voice_ratio": 0.5}

    n_declarative = 0
    n_active = 0

    for sent in sentences:
        # Declarative: does not end with ? and is not conditional
        text = sent.text.strip()
        if not text.endswith("?"):
            n_declarative += 1

        # Active voice heuristic: subject appears before main verb
        # (simplified — check if root verb has nsubj dependency)
        root = sent.root
        has_nsubj = any(child.dep_ == "nsubj" for child in root.children)
        has_nsubjpass = any(child.dep_ == "nsubjpass" for child in root.children)
        if has_nsubj and not has_nsubjpass:
            n_active += 1

    n = len(sentences)
    return {
        "declarative_ratio": n_declarative / n,
        "active_voice_ratio": n_active / n,
    }


def extract_fluency_features(doc) -> dict:
    """Extract fluency features."""
    sentences = list(doc.sents)
    n_sentences = len(sentences)

    if n_sentences == 0:
        return {"mean_sentence_length": 0.0, "n_sentences": 0, "has_repetition": False}

    lengths = [len(sent) for sent in sentences]
    mean_length = np.mean(lengths)

    # Simple repetition check: any trigram repeated
    tokens = [t.text.lower() for t in doc if not t.is_punct]
    trigrams = [" ".join(tokens[i:i+3]) for i in range(len(tokens) - 2)]
    has_repetition = len(trigrams) != len(set(trigrams))

    return {
        "mean_sentence_length": float(mean_length),
        "n_sentences": n_sentences,
        "has_repetition": has_repetition,
    }


# ---------------------------------------------------------------------------
# Composite LCS
# ---------------------------------------------------------------------------

def compute_lcs(features: LinguisticFeatures) -> float:
    """
    Compute Linguistic Confidence Score from feature vector.

    Higher LCS = the response presents its content more confidently.
    Scale: 0–1.
    """
    scores = []

    # Epistemic: more boosters relative to hedges → higher confidence
    scores.append(features.booster_ratio)

    # Evidentiality: more bald assertions → higher apparent confidence
    if features.n_sentences > 0:
        bald_ratio = features.n_bald_assertions / features.n_sentences
        scores.append(bald_ratio)
    else:
        scores.append(0.5)

    # Discourse: claim first + few concessions + definite reference → confident
    scores.append(1.0 if features.claim_in_first_sentence else 0.0)
    concessive_penalty = min(features.n_concessive_structures / 3, 1.0)
    scores.append(1.0 - concessive_penalty)
    scores.append(features.definite_reference_ratio)

    # Syntactic: declarative + active → assertive
    scores.append(features.declarative_ratio)
    scores.append(features.active_voice_ratio)

    # Fluency: longer, non-repetitive sentences → more polished/persuasive
    # (normalise sentence length to rough 0–1 scale, capping at 30 tokens)
    length_score = min(features.mean_sentence_length / 30.0, 1.0)
    scores.append(length_score)

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def extract_all_features(
    task_ids: list[str],
    domains: list[str],
    responses: list[str],
    spacy_model: str = "en_core_web_sm",
) -> list[LinguisticFeatures]:
    """Extract full linguistic feature vectors for all responses."""
    nlp = spacy.load(spacy_model)

    features_list = []

    for task_id, domain, response in tqdm(
        zip(task_ids, domains, responses), total=len(task_ids),
        desc="Extracting linguistic features"
    ):
        doc = nlp(response)

        epistemic = extract_epistemic_features(response)
        evidentiality = extract_evidentiality_features(response, doc)
        discourse = extract_discourse_features(response, doc)
        syntactic = extract_syntactic_features(doc)
        fluency = extract_fluency_features(doc)

        feats = LinguisticFeatures(
            task_id=task_id,
            domain=domain,
            **epistemic,
            **evidentiality,
            **discourse,
            **syntactic,
            **fluency,
        )
        feats.lcs_score = compute_lcs(feats)
        features_list.append(feats)

    return features_list


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_features(features: list[LinguisticFeatures], path: str | Path) -> None:
    """Save features to JSON Lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for feat in features:
            f.write(json.dumps(asdict(feat)) + "\n")
    logger.info(f"Saved {len(features)} feature vectors to {path}")
