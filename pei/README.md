# Persuasive Error Index (PEI)

**Measuring when language models are confidently wrong — and when it matters most for human oversight.**

## Motivation

Not all model errors are equal. A hedged, uncertain wrong answer is far less dangerous than a confident, well-structured one — especially when the model's internal representations encode the correct answer. The Persuasive Error Index (PEI) provides a principled framework for identifying the errors most likely to mislead human overseers, by simultaneously measuring:

1. **Internal-Surface Divergence (ISD)**: The gap between what the model's hidden states encode (probed correctness) and what it actually outputs. High ISD means the model "knows better" but says otherwise.

2. **Linguistic Confidence Score (LCS)**: A linguistically principled taxonomy of surface features — epistemic markers, evidentiality, discourse structure, syntactic assertiveness — capturing how confidently the error is presented.

3. **PEI**: A composite of ISD and LCS. High PEI errors are maximally dangerous for human-in-the-loop systems: internally known, externally confident.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run the full pipeline
python scripts/run_pipeline.py --config configs/default.yaml

# Or run individual stages
python scripts/run_pipeline.py --stage generate
python scripts/run_pipeline.py --stage judge
python scripts/run_pipeline.py --stage probe
python scripts/run_pipeline.py --stage linguistic
python scripts/run_pipeline.py --stage pei
python scripts/run_pipeline.py --stage analyse
```

## Pipeline Architecture

```
Task sets (TriviaQA, GSM8K, HellaSwag)
    │
    ▼
┌──────────────┐
│  generate.py │  → Model responses (Qwen-2.5-7B-Instruct)
└──────┬───────┘
       ▼
┌──────────────────┐
│ identify_errors.py│  → Correct/incorrect labels
└──────┬───────────┘
       │
       ├──────────────────────┐
       ▼                      ▼
┌────────────────┐   ┌──────────────────────┐
│probe_internals │   │ linguistic_features.py│
│    (ISD)       │   │       (LCS)          │
└──────┬─────────┘   └──────────┬───────────┘
       │                        │
       └────────┬───────────────┘
                ▼
         ┌────────────┐
         │ pei_score.py│  → PEI scores
         └──────┬─────┘
                ▼
         ┌────────────┐
         │ analysis.py │  → Figures, statistics, examples
         └─────────────┘
```

## Project Structure

```
pei/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── src/
│   ├── __init__.py
│   ├── generate.py            # Batch inference across task domains
│   ├── identify_errors.py     # Domain-specific answer matching
│   ├── probe_internals.py     # Activation extraction & linear probes
│   ├── linguistic_features.py # Linguistically principled LCS features
│   ├── pei_score.py           # Composite PEI computation
│   └── analysis.py            # Visualisation and statistics
├── data/
│   ├── tasks/
│   └── results/
├── notebooks/
│   └── exploration.ipynb
└── scripts/
    └── run_pipeline.py
```

## Model

Primary: **Qwen-2.5-7B-Instruct** (4-bit quantised via bitsandbytes). Chosen for its relatively strong calibration — finding high-ISD errors in a well-calibrated model is a stronger result than catching a poorly calibrated one.

## Compute Requirements

Runs on Google Colab Pro (single A100 or A10G). Estimated GPU time: 8–12 hours for inference and activation extraction. Probing and linguistic analysis run on CPU.

## Connection to AI Safety

This work supports scalable oversight by identifying where human monitoring is most likely to fail. If we deploy LLMs in high-stakes settings with human-in-the-loop oversight, we need to know which errors will actually fool the human. PEI provides a principled way to flag high-risk errors, calibrate oversight intensity, and evaluate models on a safety-relevant dimension that existing benchmarks miss.

## Author

Emmanuelle Gelain-Sohn — MSc Speech and Language Processing, University of Edinburgh.

## Licence

MIT
