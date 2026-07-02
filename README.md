# Polymath

A lightweight, locally-runnable system that routes prompts to dynamically composed LoRA adapters — each trained on a specific domain — and merges them at inference time based on the prompt's intent.

**One base model. Many specialists. One router to compose them.**

---

## How it works

```
Prompt
  │
  ▼
Router (SmolLM2-135M)
  │
  └── outputs soft weights: [α_math, α_physics, α_chem, α_bio, α_finance]
  │
  ▼
LoRA Composition: A_merged = Σ αᵢ · Aᵢ
  │
  ▼
Base Model (Qwen2.5-1.5B) + Merged Adapter
  │
  ▼
Output
```

Each domain adapter is a LoRA fine-tune of the same base model with identical rank and config, making weight-space interpolation valid. The router is a small classifier that maps a prompt to a probability distribution over domains, which then become the composition coefficients.


---

## Quickstart

```bash
git clone https://github.com/Arxane/polymath.git
cd polymath
pip install -r requirements.txt
python run.py --prompt "Explain the thermodynamics of enzyme catalysis"
```

Requires Python 3.10+, PyTorch 2.2+, and a CUDA-capable GPU. Designed to run on consumer hardware (tested on RTX 3050 6GB).

---

## Supported Domains

| Domain | Dataset |
|--------|---------|
| Math | `gsm8k`, `lighteval/MATH` |
| Physics | `camel-ai/physics` |
| Chemistry | `camel-ai/chemistry` |
| Biology | `camel-ai/biology` |
| Finance | `gbharti/finance-alpaca` |

---

## Models

| Component | Model | VRAM |
|-----------|-------|------|
| Base | `Qwen/Qwen2.5-1.5B-Instruct` | ~1.8GB |
| Router | `HuggingFaceTB/SmolLM2-135M` | ~200MB |
| Per adapter | LoRA weights only | ~20–50MB |

Total inference footprint: **~2.5GB**, comfortably within 6GB VRAM.

---

## Key Design Decisions

- **LoRA over full fine-tune** — faster per-domain training, adapter weights are tiny, composition via weighted sum is cheap
- **Soft routing over hard routing** — router outputs a distribution, not a single domain, allowing genuine multi-domain composition
- **Identical LoRA config across all adapters** — required for valid weight interpolation; mixing ranks breaks composition
- **Small router** — the router doesn't need to be smart, just intent-aware; 135M is sufficient

---

## Hardware

Developed and tested on:
- OS: Ubuntu 22.04 (WSL2)
- GPU: NVIDIA RTX 3050 6GB
- CUDA: 12.x

---

## Status

- [x] Project structure
- [ ] Domain adapter training scripts
- [ ] Router training pipeline
- [ ] Inference + composition engine
- [ ] Composition validation experiments
- [ ] Evaluation benchmarks

---

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089) — Ilharco et al., 2023
- [PEFT Library](https://github.com/huggingface/peft) — HuggingFace
