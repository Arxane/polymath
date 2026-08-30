"""
Trains a LoRA adapter for a single domain.
Usage: python scripts/train_adapter.py --domain math
"""
import argparse
import json
import yaml
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch

def load_domain_data(domain: str, max_train_samples: int | None = None) -> Dataset:
    path = Path(f"data/processed/{domain}.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"Run data/prepare_datasets.py first — {path} not found")
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    dataset = Dataset.from_list(records)

    if max_train_samples is not None and len(dataset) > max_train_samples:
        # Shuffle with a fixed seed so the subset is reproducible, without
        # mutating or truncating the underlying data/processed/*.jsonl file.
        dataset = dataset.shuffle(seed=42).select(range(max_train_samples))
        print(
            f"Capping '{domain}' training set to {max_train_samples} samples "
            f"(full dataset has {len(records)} samples)"
        )

    return dataset

def main(domain: str):
    with open("configs/base_config.yaml") as f:
        cfg = yaml.safe_load(f)

    assert domain in cfg["domains"], f"Unknown domain: {domain}"

    out_dir = Path(f"adapters/{domain}")
    if out_dir.exists():
        print(f"Adapter for '{domain}' already exists at {out_dir}. Delete it to retrain.")
        return

    print(f"\n=== Training adapter: {domain} ===\n")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    tokenizer.pad_token = tokenizer.eos_token

    # load to single GPU directly, not device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        dtype=torch.bfloat16,
        device_map={"": 0}  # force everything onto cuda:0
    )
    model.enable_input_require_grads()  # fixes the requires_grad issue

    lora_cfg = LoraConfig(
        **cfg["lora"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    max_train_samples = cfg["domains"][domain].get("max_train_samples")
    dataset = load_domain_data(domain, max_train_samples=max_train_samples)
    print(f"Dataset size: {len(dataset)} samples")

    tcfg = cfg["training"]
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(out_dir),
            dataset_text_field="text",
            num_train_epochs=int(tcfg["num_epochs"]),
            per_device_train_batch_size=int(tcfg["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(tcfg["gradient_accumulation_steps"]),
            learning_rate=float(tcfg["learning_rate"]),
            warmup_ratio=float(tcfg["warmup_ratio"]),
            max_seq_length=int(tcfg["max_seq_length"]),
            bf16=bool(tcfg["bf16"]),
            gradient_checkpointing=bool(tcfg["gradient_checkpointing"]),
            save_steps=int(tcfg["save_steps"]),
            logging_steps=int(tcfg["logging_steps"]),
            report_to="none",
        ),
    )

    trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"\nAdapter saved → {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=["math","physics","chemistry","biology","finance","psychology"])
    args = parser.parse_args()
    main(args.domain)
