"""
generate.py — Generate model responses to task sets with ground truth.

Handles loading datasets (TriviaQA, GSM8K, HellaSwag), formatting prompts,
running batch inference, and saving responses with metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TaskItem:
    """A single task with a prompt and ground-truth answer."""
    task_id: str
    domain: str          # factual_qa | reasoning | commonsense
    prompt: str
    ground_truth: str    # canonical correct answer
    metadata: dict = field(default_factory=dict)


@dataclass
class GeneratedResponse:
    """A model response paired with its task."""
    task_id: str
    domain: str
    prompt: str
    ground_truth: str
    response: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset loaders — each returns a list of TaskItems
# ---------------------------------------------------------------------------

def load_triviaqa(n_samples: int = 1500) -> list[TaskItem]:
    """Load TriviaQA (no-context) subset."""
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    items = []
    for i, row in enumerate(ds):
        # TriviaQA stores answers in answer.aliases (list of acceptable answers)
        aliases = row["answer"]["aliases"]
        canonical = row["answer"]["value"]
        items.append(TaskItem(
            task_id=f"tqa_{i:04d}",
            domain="factual_qa",
            prompt=f"Answer the following question concisely.\n\nQuestion: {row['question']}\nAnswer:",
            ground_truth=canonical,
            metadata={"aliases": aliases},
        ))
    return items


def load_gsm8k(n_samples: int = 1000) -> list[TaskItem]:
    """Load GSM8K math reasoning problems."""
    ds = load_dataset("gsm8k", "main", split="test")
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    items = []
    for i, row in enumerate(ds):
        # GSM8K answer is after #### in the solution
        answer_text = row["answer"].split("####")[-1].strip()
        items.append(TaskItem(
            task_id=f"gsm_{i:04d}",
            domain="reasoning",
            prompt=(
                "Solve the following maths problem. "
                "Show your working, then give the final numerical answer "
                "after 'The answer is'.\n\n"
                f"Problem: {row['question']}\nSolution:"
            ),
            ground_truth=answer_text,
            metadata={"full_solution": row["answer"]},
        ))
    return items


def load_hellaswag(n_samples: int = 1000) -> list[TaskItem]:
    """Load HellaSwag commonsense completion task."""
    ds = load_dataset("Rowan/hellaswag", split="validation")
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    items = []
    for i, row in enumerate(ds):
        label = int(row["label"])
        endings = row["endings"]
        correct_ending = endings[label]

        # Format as a multiple-choice prompt
        choices_str = "\n".join(
            f"  ({chr(65 + j)}) {e}" for j, e in enumerate(endings)
        )
        items.append(TaskItem(
            task_id=f"hs_{i:04d}",
            domain="commonsense",
            prompt=(
                "Choose the most plausible continuation.\n\n"
                f"Context: {row['ctx']}\n\nOptions:\n{choices_str}\n\n"
                "Answer with the letter only (A, B, C, or D):"
            ),
            ground_truth=chr(65 + label),
            metadata={"correct_ending": correct_ending, "label_idx": label},
        ))
    return items


LOADERS = {
    "factual_qa": load_triviaqa,
    "reasoning": load_gsm8k,
    "commonsense": load_hellaswag,
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(config: dict) -> tuple:
    """Load model and tokeniser with optional 4-bit quantisation."""
    model_name = config["model"]["name"]
    quant = config["model"].get("quantisation", None)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"device_map": config["model"].get("device", "auto")}

    if quant == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

    logger.info(f"Loading {model_name} (quantisation={quant})")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def clean_response(text: str) -> str:
    """Strip system prompt artefacts and trailing junk from generated text."""
    # Common system prompt leaks from instruction-tuned models
    cutoff_phrases = [
        "You are an AI assistant",
        "You are a helpful assistant",
        "<|im_end|>",
        "<|im_start|>",
        "<|endoftext|>",
    ]
    for phrase in cutoff_phrases:
        idx = text.find(phrase)
        if idx > 0:  # only cut if it's not at the very start
            text = text[:idx]
    return text.strip()


def format_chat_prompt(tokenizer, prompt: str) -> str:
    """Format prompt using the model's chat template."""
    messages = [{"role": "user", "content": prompt}]
    # apply_chat_template returns token IDs or a string depending on args
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return formatted


def generate_responses(
    model,
    tokenizer,
    tasks: list[TaskItem],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    batch_size: int = 8,
) -> list[GeneratedResponse]:
    """Run batch inference over tasks and return responses."""
    responses = []

    for start in tqdm(range(0, len(tasks), batch_size), desc="Generating"):
        batch = tasks[start : start + batch_size]

        # Use chat template for proper formatting
        formatted_prompts = [format_chat_prompt(tokenizer, t.prompt) for t in batch]

        inputs = tokenizer(
            formatted_prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                pad_token_id=tokenizer.pad_token_id,
            )
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

            outputs = model.generate(**gen_kwargs)

        # Decode only the generated portion
        for j, task in enumerate(batch):
            input_len = inputs["input_ids"][j].shape[0]
            generated_ids = outputs[j][input_len:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            text = clean_response(text)

            responses.append(GeneratedResponse(
                task_id=task.task_id,
                domain=task.domain,
                prompt=task.prompt,
                ground_truth=task.ground_truth,
                response=text,
                metadata=task.metadata,
            ))

    return responses


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_responses(responses: list[GeneratedResponse], path: str | Path) -> None:
    """Save responses to JSON Lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in responses:
            f.write(json.dumps(asdict(r)) + "\n")
    logger.info(f"Saved {len(responses)} responses to {path}")


def load_responses(path: str | Path) -> list[GeneratedResponse]:
    """Load responses from JSON Lines."""
    responses = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            responses.append(GeneratedResponse(**d))
    return responses


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_generation(config_path: str = "configs/default.yaml") -> list[GeneratedResponse]:
    """Full generation pipeline: load data → load model → generate → save."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load tasks from all configured domains
    all_tasks = []
    for domain, domain_cfg in config["tasks"].items():
        loader = LOADERS.get(domain)
        if loader is None:
            logger.warning(f"No loader for domain '{domain}', skipping")
            continue
        tasks = loader(n_samples=domain_cfg["n_samples"])
        all_tasks.extend(tasks)
        logger.info(f"Loaded {len(tasks)} tasks for {domain}")

    logger.info(f"Total tasks: {len(all_tasks)}")

    # Load model
    model, tokenizer = load_model(config)

    # Generate
    gen_cfg = config["generation"]
    responses = generate_responses(
        model, tokenizer, all_tasks,
        max_new_tokens=gen_cfg["max_new_tokens"],
        temperature=gen_cfg["temperature"],
        batch_size=gen_cfg["batch_size"],
    )

    # Save
    out_path = Path(config["output"]["results_dir"]) / "responses.jsonl"
    save_responses(responses, out_path)

    return responses
