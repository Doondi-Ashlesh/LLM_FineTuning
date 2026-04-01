# LLM Fine-Tuning with Unsloth

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Framework](https://img.shields.io/badge/Framework-Unsloth-orange)
![Technique](https://img.shields.io/badge/Technique-LoRA%20%7C%20QLoRA-green)
![GPU](https://img.shields.io/badge/GPU-CUDA%207.5%2B-76b900?logo=nvidia&logoColor=white)
![Models](https://img.shields.io/badge/Models-Llama--3.1--8B%20%7C%20Phi--3-blueviolet)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A collection of parameter-efficient LLM fine-tuning experiments using **Unsloth**, **LoRA**, and **QLoRA** — covering REST API knowledge injection, resume generation, and Text-to-SQL translation.

---

## Overview

This repository demonstrates end-to-end LLM fine-tuning workflows built on top of the [Unsloth](https://github.com/unslothai/unsloth) framework for fast, memory-efficient training. Each project targets a different domain and showcases how small, domain-specific datasets can meaningfully improve large language model behavior using parameter-efficient fine-tuning (PEFT) techniques.

Key themes across all projects:
- **4-bit quantization** via `bitsandbytes` for GPU memory efficiency
- **LoRA / QLoRA** with consistent hyperparameter configurations
- **Structured evaluation** comparing base vs. fine-tuned model outputs

---

## Projects

| # | Project | Model | Technique | Domain |
|---|---------|-------|-----------|--------|
| 1 | [Basic LoRA Fine-Tuning](#1-basic-lora-fine-tuning) | Llama-3.1-8B | LoRA | REST API Knowledge |
| 2 | [Resume Bullet Optimizer](#2-resume-bullet-optimizer) | Llama-3.1-8B | LoRA | Resume Writing |
| 3 | [Phi-3 Text-to-SQL](#3-phi-3-text-to-sql-with-qlora) | Phi-3 | QLoRA | SQL Generation |

---

## 1. Basic LoRA Fine-Tuning

**Notebook:** [`LLM_Finetuning_UnSloth_Basic.ipynb`](./LLM_Finetuning_UnSloth_Basic.ipynb)

### Goal
A proof-of-concept demonstrating the full LoRA fine-tuning pipeline on a minimal custom dataset about REST APIs.

### Setup

| Parameter | Value |
|-----------|-------|
| Base Model | `meta-llama/Meta-Llama-3.1-8B` |
| Quantization | 4-bit (bitsandbytes) |
| LoRA Rank (`r`) | 16 |
| LoRA Alpha | 16 |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Trainable Parameters | 13.6M / 8.04B (0.17%) |
| Max Sequence Length | 2048 |
| Training Steps | 30 |
| Learning Rate | 2e-4 |

### Dataset
4 custom instruction-output pairs covering REST API concepts (explanations, use cases, GET vs POST, request mechanics). Formatted using Unsloth's `alpaca` chat template.

### Results

| Metric | Value |
|--------|-------|
| Final Training Loss | 1.010 |
| Training Runtime | 36.3 seconds |
| Samples/sec | 6.6 |

**Sample inference:**
```
Input:  "What is the difference between REST and GraphQL?"
Output: "REST sends simple requests to URLs, while GraphQL sends a single
         request that allows a client to fetch or send data in a simple way."
```

---

## 2. Resume Bullet Optimizer

**Notebook:** [`Resume_FineTuning/Resume_LLM_FineTuning_UnSloth_fixed.ipynb`](./Resume_FineTuning/Resume_LLM_FineTuning_UnSloth_fixed.ipynb)

### Goal
Fine-tune Llama-3.1-8B to transform weak, generic resume bullet points into strong, impact-driven software engineering accomplishments.

### Setup

| Parameter | Value |
|-----------|-------|
| Base Model | `meta-llama/Meta-Llama-3.1-8B` |
| Quantization | 4-bit (bitsandbytes) |
| LoRA Rank (`r`) | 16 |
| LoRA Alpha | 16 |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Training Steps | 120 |
| Learning Rate | 2e-4 |
| Eval Steps | Every 10 steps |
| Random Seed | 42 |

### Dataset
- **Training:** 30 instruction-input-output triplets across SE, AI, Cloud, and Frontend roles
- **Evaluation:** Held-out examples for base vs. fine-tuned comparison

**Sample pair:**
```
Input:   "Built APIs for student platform."
Output:  "Developed scalable backend APIs for a student platform, improving
          service reliability and supporting high-volume user workflows."
```

### Evaluation
Outputs were scored manually on 4 dimensions:

| Dimension | Description |
|-----------|-------------|
| **Strength** | Impact and action-orientation of the bullet |
| **Clarity** | Readability and precision |
| **Conciseness** | No filler words or redundancy |
| **Realism** | Plausibility as a real resume bullet |

Results across 44 evaluation examples are saved in:
- [`base_vs_finetuned_comparison.csv`](./Resume_FineTuning/base_vs_finetuned_comparison.csv) — raw model outputs
- [`base_vs_finetuned_comparison_rated.csv`](./Resume_FineTuning/base_vs_finetuned_comparison_rated.csv) — manually rated comparison

---

## 3. Phi-3 Text-to-SQL with QLoRA

**Directory:** [`phi_3_Text2Sql_QLoRA/`](./phi_3_Text2Sql_QLoRA/)

### Goal
Fine-tune Microsoft's Phi-3 model on a Text-to-SQL task using QLoRA — enabling the model to translate natural language questions into executable SQL queries given a table schema.

### Technique
**QLoRA** (Quantized LoRA) — combines 4-bit NF4 quantization with low-rank adapter training, enabling fine-tuning with significantly reduced GPU memory compared to standard LoRA.

### Dataset
208 examples of natural language question + table schema → SQL query pairs.

**Sample:**
```
Question: "Which player had a To par of 13?"
Schema:   Players table with columns: name, score, to_par, rank, ...
Output:   SELECT name FROM players WHERE to_par = '13'
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `exact_match` | Exact string match with ground truth SQL |
| `parses` | SQL syntax validity (parseable by SQL parser) |
| `strict_match` | Normalized canonical SQL match |
| `canonical_match` | Semantic equivalence after normalization |
| `error_type` | Failure classification (`column_mismatch`, `invalid_sql`, etc.) |

### Results

Evaluation outputs are in:
- [`phi3_text2sql_comparison_v2.csv`](./phi_3_Text2Sql_QLoRA/phi3_text2sql_comparison_v2.csv) — full comparison (208 examples, all metrics)
- [`phi3_text2sql_ft_beats_base_v2.csv`](./phi_3_Text2Sql_QLoRA/phi3_text2sql_ft_beats_base_v2.csv) — 177 examples where fine-tuned outperformed base
- [`phi3_text2sql_ft_eval.csv`](./phi_3_Text2Sql_QLoRA/phi3_text2sql_ft_eval.csv) — 100-example validation subset

Key observations:
- Fine-tuned model significantly improves quote handling (single vs. double quotes in SQL)
- Reduces `invalid_sql` errors vs. base model
- 177/208 examples (85%) show improvement over base Phi-3

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| [Unsloth](https://github.com/unslothai/unsloth) | Fast LoRA/QLoRA fine-tuning engine |
| [Transformers](https://github.com/huggingface/transformers) | Model loading, tokenization, inference |
| [TRL](https://github.com/huggingface/trl) | `SFTTrainer` for supervised fine-tuning |
| [PEFT](https://github.com/huggingface/peft) | LoRA adapter management |
| [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | 4-bit quantization |
| [Datasets](https://github.com/huggingface/datasets) | Dataset loading and formatting |
| [xformers](https://github.com/facebookresearch/xformers) | Memory-efficient attention |
| [Accelerate](https://github.com/huggingface/accelerate) | Distributed/mixed-precision training |
| pandas | Evaluation result analysis |

---

## Hardware Requirements

These notebooks were developed and tested on:

| Requirement | Minimum | Tested On |
|-------------|---------|-----------|
| GPU | 12 GB VRAM | NVIDIA H100 / RTX Pro 6000 (Blackwell) |
| CUDA Version | 7.5+ | CUDA 12.x |
| RAM | 16 GB | — |
| Python | 3.10+ | 3.10 |

> Experiments were conducted on an **NVIDIA H100** and **NVIDIA RTX Pro 6000 (Blackwell edition)**. 4-bit quantization is still used for memory efficiency and faster training. For consumer GPUs (8–16 GB VRAM), reduce `max_seq_length` or `per_device_train_batch_size`.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/LLM_FineTuning.git
cd LLM_FineTuning
```

### 2. Install Dependencies

Install Unsloth and required packages (run inside the notebook or a virtual environment):

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes xformers
pip install transformers datasets pandas
```

> For Colab environments, Unsloth auto-detects the CUDA version and installs the correct wheels.

### 3. Authenticate with Hugging Face

Llama-3.1-8B is a gated model. You need to:
1. Accept the model license at [meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
2. Set your HF token:

```python
from huggingface_hub import login
login(token="hf_your_token_here")
```

### 4. Run a Notebook

Open any of the notebooks in Jupyter or Google Colab and run all cells:

```bash
jupyter notebook LLM_Finetuning_UnSloth_Basic.ipynb
```

---

## Repository Structure

```
LLM_FineTuning/
│
├── LLM_Finetuning_UnSloth_Basic.ipynb          # Project 1: Basic LoRA demo (Llama-3.1-8B)
│
├── Resume_FineTuning/
│   ├── Resume_LLM_FineTuning_UnSloth_fixed.ipynb   # Project 2: Resume bullet optimizer
│   ├── base_vs_finetuned_comparison.csv             # Raw model output comparison
│   └── base_vs_finetuned_comparison_rated.csv       # Manually scored results
│
└── phi_3_Text2Sql_QLoRA/
    ├── phi3_text2sql_comparison.csv                 # Base comparison (v1 metrics)
    ├── phi3_text2sql_comparison_v2.csv              # Full comparison (v2 metrics, 208 rows)
    ├── phi3_text2sql_ft_beats_base_v2.csv           # FT > Base cases (177 rows)
    ├── phi3_text2sql_ft_eval.csv                    # FT eval subset (100 rows)
    ├── phi3_text2sql_base_recovered_v2.csv          # Base model recovered outputs
    └── phi3_text2sql_ft_recovered_v2.csv            # FT model recovered outputs
```

---

## Results Summary

| Project | Model | Dataset Size | Key Result |
|---------|-------|-------------|------------|
| Basic LoRA | Llama-3.1-8B | 4 examples | Training loss: 1.010 in 36s |
| Resume Optimizer | Llama-3.1-8B | 30 train / 44 eval | Improved strength, clarity, conciseness vs. base |
| Text-to-SQL | Phi-3 + QLoRA | 208 examples | 85% of eval examples: FT beats base |

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-experiment`)
3. Commit your changes (`git commit -m 'Add new fine-tuning experiment'`)
4. Push to the branch (`git push origin feature/new-experiment`)
5. Open a Pull Request
