"""
run_pipeline.py — End-to-end PEI pipeline.

Usage:
    python scripts/run_pipeline.py [--config configs/default.yaml] [--stage STAGE]

Stages:
    generate    — Generate model responses
    judge       — Identify errors
    probe       — Extract activations and train probes (ISD)
    linguistic  — Extract linguistic features (LCS)
    pei         — Compute PEI scores
    analyse     — Run analysis and produce figures
    all         — Run everything (default)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generate import run_generation, load_responses, LOADERS, load_model, format_chat_prompt
from src.identify_errors import judge_responses, save_judged, load_judged
from src.probe_internals import (
    select_layers, extract_activations, train_probes, compute_isd,
    save_activations, save_probes, save_isd_scores,
)
from src.linguistic_features import extract_all_features, save_features
from src.pei_score import compute_pei, save_pei, load_pei
from src.analysis import (
    pei_results_to_df, plot_pei_distribution, plot_isd_vs_lcs,
    plot_probe_accuracy, showcase_examples, compute_summary_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="PEI Pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--stage", default="all",
        choices=["generate", "judge", "probe", "linguistic", "pei", "analyse", "all"],
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = Path(config["output"]["results_dir"])
    figures_dir = Path(config["output"]["figures_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    stages = (
        ["generate", "judge", "probe", "linguistic", "pei", "analyse"]
        if args.stage == "all"
        else [args.stage]
    )

    # ------------------------------------------------------------------
    # Stage: generate
    # ------------------------------------------------------------------
    if "generate" in stages:
        logger.info("=" * 60)
        logger.info("STAGE: Generate model responses")
        logger.info("=" * 60)
        responses = run_generation(args.config)
        logger.info(f"Generated {len(responses)} responses")

    # ------------------------------------------------------------------
    # Stage: judge
    # ------------------------------------------------------------------
    if "judge" in stages:
        logger.info("=" * 60)
        logger.info("STAGE: Judge responses (identify errors)")
        logger.info("=" * 60)
        responses = load_responses(results_dir / "responses.jsonl")
        judged = judge_responses(responses)
        save_judged(judged, results_dir / "judged.jsonl")

    # ------------------------------------------------------------------
    # Stage: probe (ISD)
    # ------------------------------------------------------------------
    if "probe" in stages:
        logger.info("=" * 60)
        logger.info("STAGE: Extract activations and train probes (ISD)")
        logger.info("=" * 60)
        judged = load_judged(results_dir / "judged.jsonl")

        # Load model for activation extraction
        model, tokenizer = load_model(config)
        n_layers = model.config.num_hidden_layers
        layer_indices = select_layers(n_layers)
        logger.info(f"Probing layers: {layer_indices} (of {n_layers} total)")

        # Extract activations
        prompts = [j.prompt for j in judged]
        responses_text = [j.response for j in judged]
        activations = extract_activations(
            model, tokenizer, prompts, responses_text,
            layer_indices=layer_indices,
            batch_size=config["generation"]["batch_size"] // 2,  # smaller for memory
            format_fn=format_chat_prompt,
        )
        save_activations(activations, results_dir / "activations.npz")

        # Train probes
        labels = np.array([1 if j.is_correct else 0 for j in judged])
        probes, probe_results = train_probes(
            activations, labels,
            test_split=config["probing"]["test_split"],
            random_seed=config["probing"]["random_seed"],
        )
        save_probes(probes, results_dir / "probes.pkl")

        # Save probe results
        from dataclasses import asdict
        with open(results_dir / "probe_results.json", "w") as f:
            json.dump([asdict(r) for r in probe_results], f, indent=2)

        # Compute ISD
        task_ids = [j.task_id for j in judged]
        domains = [j.domain for j in judged]
        isd_scores = compute_isd(activations, probes, task_ids, domains, labels)
        save_isd_scores(isd_scores, results_dir / "isd_scores.jsonl")

    # ------------------------------------------------------------------
    # Stage: linguistic (LCS)
    # ------------------------------------------------------------------
    if "linguistic" in stages:
        logger.info("=" * 60)
        logger.info("STAGE: Extract linguistic features (LCS)")
        logger.info("=" * 60)
        judged = load_judged(results_dir / "judged.jsonl")

        features = extract_all_features(
            task_ids=[j.task_id for j in judged],
            domains=[j.domain for j in judged],
            responses=[j.response for j in judged],
            spacy_model=config["linguistic_features"]["spacy_model"],
        )
        save_features(features, results_dir / "linguistic_features.jsonl")

    # ------------------------------------------------------------------
    # Stage: pei
    # ------------------------------------------------------------------
    if "pei" in stages:
        logger.info("=" * 60)
        logger.info("STAGE: Compute PEI scores")
        logger.info("=" * 60)
        from src.probe_internals import ISDScore
        from src.linguistic_features import LinguisticFeatures

        # Load ISD and LCS
        isd_scores = []
        with open(results_dir / "isd_scores.jsonl") as f:
            for line in f:
                d = json.loads(line)
                isd_scores.append(ISDScore(**d))

        lcs_features = []
        with open(results_dir / "linguistic_features.jsonl") as f:
            for line in f:
                d = json.loads(line)
                lcs_features.append(LinguisticFeatures(**d))

        pei_cfg = config["pei"]
        pei_results = compute_pei(
            isd_scores, lcs_features,
            isd_weight=pei_cfg["isd_weight"],
            lcs_weight=pei_cfg["lcs_weight"],
            normalisation=pei_cfg["normalisation"],
        )
        save_pei(pei_results, results_dir / "pei_scores.jsonl")

    # ------------------------------------------------------------------
    # Stage: analyse
    # ------------------------------------------------------------------
    if "analyse" in stages:
        logger.info("=" * 60)
        logger.info("STAGE: Analysis and visualisation")
        logger.info("=" * 60)
        pei_results = load_pei(results_dir / "pei_scores.jsonl")
        df = pei_results_to_df(pei_results)

        # Plots
        plot_pei_distribution(df, figures_dir / "pei_distribution.png")
        plot_isd_vs_lcs(df, figures_dir / "isd_vs_lcs.png")

        # Probe results
        if (results_dir / "probe_results.json").exists():
            from src.probe_internals import ProbeResult
            with open(results_dir / "probe_results.json") as f:
                probe_data = json.load(f)
            probe_results = [ProbeResult(**d) for d in probe_data]
            plot_probe_accuracy(probe_results, figures_dir / "probe_accuracy.png")

        # Examples
        if (results_dir / "judged.jsonl").exists():
            judged = load_judged(results_dir / "judged.jsonl")
            examples = showcase_examples(pei_results, judged, n_examples=5)
            with open(results_dir / "example_showcase.json", "w") as f:
                json.dump(examples, f, indent=2)

        # Summary
        summary = compute_summary_stats(df)
        with open(results_dir / "summary_stats.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary: {json.dumps(summary, indent=2)}")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
