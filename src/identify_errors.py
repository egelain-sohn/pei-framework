"""
identify_errors.py — Compare model responses to ground truth.

Handles domain-specific answer extraction and matching:
  - Factual QA: fuzzy string matching against answer aliases
  - Reasoning (GSM8K): numerical extraction and comparison
  - Commonsense (HellaSwag): letter matching
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class JudgedResponse:
    """A response with a correctness judgement."""
    task_id: str
    domain: str
    prompt: str
    ground_truth: str
    response: str
    is_correct: bool
    extracted_answer: str  # what we parsed as the model's answer
    metadata: dict


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_factual_answer(response: str) -> str:
    """Extract the core answer from a factual QA response."""
    # Take the first line / sentence as the answer
    answer = response.strip().split("\n")[0].strip()
    # Remove common preambles
    for prefix in ["The answer is", "Answer:", "A:"]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    # Strip trailing punctuation
    answer = answer.rstrip(".")
    return answer


def extract_numerical_answer(response: str) -> str:
    """Extract the final numerical answer from a GSM8K-style response."""
    # Look for 'The answer is X' pattern first
    match = re.search(r"[Tt]he answer is\s*[:\s]*\$?([\d,]+(?:\.\d+)?)", response)
    if match:
        return match.group(1).replace(",", "")

    # Fallback: last number in the response
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", response)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def extract_letter_answer(response: str) -> str:
    """Extract a multiple-choice letter (A/B/C/D) from a response."""
    response_clean = response.strip().upper()
    # Direct single letter
    if response_clean in ("A", "B", "C", "D"):
        return response_clean
    # Look for patterns like (A), A), A.
    match = re.search(r"\(?([A-D])\)?[.\s:]", response_clean)
    if match:
        return match.group(1)
    # First letter if response starts with one
    if response_clean and response_clean[0] in "ABCD":
        return response_clean[0]
    return ""


EXTRACTORS = {
    "factual_qa": extract_factual_answer,
    "reasoning": extract_numerical_answer,
    "commonsense": extract_letter_answer,
}


# ---------------------------------------------------------------------------
# Answer matching
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Lowercase, strip articles and punctuation for fuzzy matching."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def match_factual(extracted: str, ground_truth: str, aliases: list[str] = None) -> bool:
    """Check if extracted answer matches any acceptable answer."""
    norm_ext = normalise(extracted)
    candidates = [ground_truth] + (aliases or [])

    for candidate in candidates:
        norm_cand = normalise(candidate)
        # Exact match after normalisation
        if norm_ext == norm_cand:
            return True
        # Containment (either direction) for short answers
        if len(norm_cand) > 2 and (norm_cand in norm_ext or norm_ext in norm_cand):
            return True

    return False


def match_numerical(extracted: str, ground_truth: str) -> bool:
    """Check if extracted number matches ground truth numerically."""
    try:
        return float(extracted) == float(ground_truth.replace(",", ""))
    except (ValueError, TypeError):
        return False


def match_letter(extracted: str, ground_truth: str) -> bool:
    """Check if extracted letter matches ground truth."""
    return extracted.upper().strip() == ground_truth.upper().strip()


MATCHERS = {
    "factual_qa": match_factual,
    "reasoning": match_numerical,
    "commonsense": match_letter,
}


# ---------------------------------------------------------------------------
# Main judgement pipeline
# ---------------------------------------------------------------------------

def judge_responses(responses: list) -> list[JudgedResponse]:
    """Judge all responses for correctness."""
    judged = []

    for r in responses:
        extractor = EXTRACTORS.get(r.domain, extract_factual_answer)
        matcher = MATCHERS.get(r.domain, match_factual)

        extracted = extractor(r.response)

        # Domain-specific matching
        if r.domain == "factual_qa":
            is_correct = matcher(extracted, r.ground_truth, r.metadata.get("aliases", []))
        else:
            is_correct = matcher(extracted, r.ground_truth)

        judged.append(JudgedResponse(
            task_id=r.task_id,
            domain=r.domain,
            prompt=r.prompt,
            ground_truth=r.ground_truth,
            response=r.response,
            is_correct=is_correct,
            extracted_answer=extracted,
            metadata=r.metadata,
        ))

    # Summary statistics
    total = len(judged)
    correct = sum(1 for j in judged if j.is_correct)
    errors = total - correct
    logger.info(
        f"Judged {total} responses: {correct} correct ({correct/total:.1%}), "
        f"{errors} errors ({errors/total:.1%})"
    )
    for domain in set(j.domain for j in judged):
        domain_items = [j for j in judged if j.domain == domain]
        domain_correct = sum(1 for j in domain_items if j.is_correct)
        logger.info(
            f"  {domain}: {domain_correct}/{len(domain_items)} correct "
            f"({domain_correct/len(domain_items):.1%})"
        )

    return judged


def save_judged(judged: list[JudgedResponse], path: str | Path) -> None:
    """Save judged responses to JSON Lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for j in judged:
            f.write(json.dumps(asdict(j)) + "\n")
    logger.info(f"Saved {len(judged)} judged responses to {path}")


def load_judged(path: str | Path) -> list[JudgedResponse]:
    """Load judged responses from JSON Lines."""
    judged = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            judged.append(JudgedResponse(**d))
    return judged
