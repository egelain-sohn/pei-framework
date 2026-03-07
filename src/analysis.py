"""
analysis.py — Visualisation and statistical analysis of PEI results.

Produces:
  - PEI distribution plots (overall and by domain)
  - ISD vs LCS scatter plots
  - Probe accuracy per layer
  - Example showcases (high-PEI vs low-PEI errors)
  - Summary statistics
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

# Consistent styling
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLOURS = {"factual_qa": "#4C72B0", "reasoning": "#DD8452", "commonsense": "#55A868"}


def pei_results_to_df(results: list) -> pd.DataFrame:
    """Convert PEI results to a DataFrame."""
    from dataclasses import asdict
    return pd.DataFrame([asdict(r) for r in results])


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------

def plot_pei_distribution(df: pd.DataFrame, save_path: str | Path = None) -> None:
    """Plot PEI score distribution, overall and by domain."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall distribution
    axes[0].hist(df["pei_score"], bins=30, edgecolor="white", alpha=0.8, color="#4C72B0")
    axes[0].set_xlabel("PEI Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("PEI Distribution (All Errors)")
    axes[0].axvline(df["pei_score"].mean(), color="red", linestyle="--", label="Mean")
    axes[0].legend()

    # By domain
    for domain in df["domain"].unique():
        subset = df[df["domain"] == domain]
        axes[1].hist(
            subset["pei_score"], bins=20, alpha=0.6,
            label=domain, color=COLOURS.get(domain, None),
        )
    axes[1].set_xlabel("PEI Score")
    axes[1].set_ylabel("Count")
    axes[1].set_title("PEI Distribution by Domain")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved PEI distribution plot to {save_path}")
    plt.close()


def plot_isd_vs_lcs(df: pd.DataFrame, save_path: str | Path = None) -> None:
    """Scatter plot of ISD vs LCS, coloured by domain."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for domain in df["domain"].unique():
        subset = df[df["domain"] == domain]
        ax.scatter(
            subset["isd_score"], subset["lcs_score"],
            alpha=0.5, label=domain, color=COLOURS.get(domain, None),
            s=30,
        )

    ax.set_xlabel("ISD Score (internal confidence in correct answer)")
    ax.set_ylabel("LCS Score (linguistic confidence of presentation)")
    ax.set_title("Internal-Surface Divergence vs Linguistic Confidence")
    ax.legend()

    # Annotate quadrants
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5)
    ax.axvline(0.5, color="grey", linestyle=":", alpha=0.5)
    ax.text(0.75, 0.9, "HIGH PEI\n(most dangerous)", transform=ax.transAxes,
            ha="center", va="center", fontsize=9, color="red", alpha=0.7)
    ax.text(0.25, 0.1, "LOW PEI\n(least dangerous)", transform=ax.transAxes,
            ha="center", va="center", fontsize=9, color="green", alpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved ISD vs LCS plot to {save_path}")
    plt.close()


def plot_probe_accuracy(probe_results: list, save_path: str | Path = None) -> None:
    """Bar chart of probe accuracy and AUC per layer."""
    from dataclasses import asdict
    df = pd.DataFrame([asdict(r) for r in probe_results])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width/2, df["accuracy"], width, label="Accuracy", color="#4C72B0")
    ax.bar(x + width/2, df["auc"], width, label="AUC", color="#DD8452")

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Score")
    ax.set_title("Linear Probe Performance by Layer")
    ax.set_xticks(x)
    ax.set_xticklabels(df["layer_idx"])
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5, label="Chance")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved probe accuracy plot to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Example showcases
# ---------------------------------------------------------------------------

def showcase_examples(
    pei_results: list,
    judged_responses: list,
    n_examples: int = 5,
) -> dict:
    """
    Select high-PEI and low-PEI error examples for qualitative analysis.

    Returns dict with 'high_pei' and 'low_pei' lists.
    """
    from dataclasses import asdict

    response_lookup = {j.task_id: j for j in judged_responses}

    # Sort by PEI
    sorted_results = sorted(pei_results, key=lambda r: r.pei_score, reverse=True)
    errors_only = [r for r in sorted_results if not r.is_correct]

    high_pei = errors_only[:n_examples]
    low_pei = errors_only[-n_examples:]

    def format_example(pei_result):
        resp = response_lookup.get(pei_result.task_id)
        if resp is None:
            return asdict(pei_result)
        return {
            "task_id": pei_result.task_id,
            "domain": pei_result.domain,
            "pei_score": round(pei_result.pei_score, 3),
            "isd_score": round(pei_result.isd_score, 3),
            "lcs_score": round(pei_result.lcs_score, 3),
            "prompt": resp.prompt[:200] + "..." if len(resp.prompt) > 200 else resp.prompt,
            "response": resp.response[:300] + "..." if len(resp.response) > 300 else resp.response,
            "ground_truth": resp.ground_truth,
            "extracted_answer": resp.extracted_answer,
        }

    return {
        "high_pei": [format_example(r) for r in high_pei],
        "low_pei": [format_example(r) for r in low_pei],
    }


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary_stats(df: pd.DataFrame) -> dict:
    """Compute summary statistics for the PEI results."""
    summary = {
        "n_errors": len(df),
        "pei_mean": float(df["pei_score"].mean()),
        "pei_std": float(df["pei_score"].std()),
        "pei_median": float(df["pei_score"].median()),
        "isd_lcs_correlation": float(df["isd_score"].corr(df["lcs_score"])),
    }

    # Per-domain statistics
    for domain in df["domain"].unique():
        subset = df[df["domain"] == domain]
        summary[f"{domain}_n"] = len(subset)
        summary[f"{domain}_pei_mean"] = float(subset["pei_score"].mean())
        summary[f"{domain}_pei_std"] = float(subset["pei_score"].std())

    # Test whether PEI differs across domains (Kruskal-Wallis)
    groups = [subset["pei_score"].values for _, subset in df.groupby("domain")]
    if len(groups) > 1:
        stat, p = stats.kruskal(*groups)
        summary["domain_kruskal_h"] = float(stat)
        summary["domain_kruskal_p"] = float(p)

    return summary
