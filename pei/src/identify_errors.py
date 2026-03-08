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
    """Extract the core answer from a factual QA response.

    Strategy (in priority order):
      1. If the model explicitly restates with a preamble like
         "To provide a concise answer:", prefer that line.
      2. Otherwise, take the first non-empty line.
      3. Strip common answer preambles.
      4. If the result is still a full sentence (>6 words), try to
         isolate the entity after a copula.
    The *matcher* enforces a length check on top of this, so returning
    a longish string is acceptable — it just won't get containment
    matching.
    """
    text = response.strip()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    # Check for an explicit restatement line anywhere in the response
    restatement_cues = [
        "to provide a concise answer",
        "in short",
        "to summarise",
        "to summarize",
        "the short answer is",
        "simply put",
        "concisely",
    ]
    for line in lines:
        line_lower = line.lower()
        for cue in restatement_cues:
            if cue in line_lower:
                # Take everything after the cue
                idx = line_lower.index(cue) + len(cue)
                after = line[idx:].lstrip(":,; ").strip().rstrip(".!,;:")
                if after:
                    text = after
                    lines = None  # signal that we've found our answer
                    break
        if lines is None:
            break

    # If no restatement found, take the first non-empty line
    if lines is not None:
        text = lines[0] if lines else text

    # Strip common answer preambles (case-insensitive)
    preambles = [
        "the answer is", "answer:", "a:", "the answer to this question is",
        "the answer to the question is", "the answer to that is",
    ]
    text_lower = text.lower()
    for prefix in preambles:
        if text_lower.startswith(prefix):
            text = text[len(prefix):].strip()
            text_lower = text.lower()

    # Strip trailing punctuation
    text = text.rstrip(".!,;:")

    # If still a full sentence (>6 words), try to extract what follows a copula
    words = text.split()
    if len(words) > 6:
        for splitter in [" is ", " was ", " are ", " were "]:
            if splitter in text.lower():
                idx = text.lower().index(splitter) + len(splitter)
                candidate = text[idx:].strip().rstrip(".")
                if 1 <= len(candidate.split()) <= len(words) - 2:
                    text = candidate
                break

    return text.strip()


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
    # Look for patterns like (A), A), A., "A "
    match = re.search(r"\(?([A-D])\)?[.\s:,)]", response_clean)
    if match:
        return match.group(1)
    # First character — but only if it's a letter followed by a
    # non-alphabetic character (so "ALTHOUGH..." doesn't match as "A")
    if len(response_clean) >= 2 and response_clean[0] in "ABCD" and not response_clean[1].isalpha():
        return response_clean[0]
    if len(response_clean) == 1 and response_clean in "ABCD":
        return response_clean
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
    """Check if extracted answer matches any acceptable answer.

    Three tiers of matching, applied in order:
      1. Exact normalised equality — always.
      2. Containment — only when the extracted string is concise (≤5
         normalised tokens), preventing spurious hits in long sentences.
      3. Suffix-anchored — when the ground truth is short (≤3 tokens)
         and appears at the very end of the extracted string.  This
         catches cases like extracted="killed in 1934", gt="1934".
    """
    norm_ext = normalise(extracted)
    ext_words = norm_ext.split()
    candidates = [ground_truth] + (aliases or [])

    for candidate in candidates:
        norm_cand = normalise(candidate)
        if not norm_cand:
            continue
        cand_words = norm_cand.split()

        # Tier 1: exact match after normalisation
        if norm_ext == norm_cand:
            return True

        # Tier 2: containment — only when extracted is short
        if len(ext_words) <= 5 and len(norm_cand) > 2:
            if norm_cand in norm_ext or norm_ext in norm_cand:
                return True

        # Tier 3: suffix-anchored — short ground truth at end of extraction
        if len(cand_words) <= 3 and len(ext_words) > len(cand_words):
            if ext_words[-len(cand_words):] == cand_words:
                return True

        # Tier 4: single-word ground truth appearing as a standalone word
        # in the extraction.  Catches "thick pea soup" matching gt "soup".
        # Limited to 1-word ground truths to avoid spurious partial matches.
        if len(cand_words) == 1 and norm_cand in ext_words:
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
