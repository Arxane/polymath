import json
import yaml
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def format_sample(input_text, output_text):
    return f"### Instruction:\n{input_text.strip()}\n\n### Response:\n{output_text.strip()}"

def prepare(cfg):
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    for domain, dcfg in cfg["domains"].items():
        out_path = out_dir / f"{domain}.jsonl"
        if out_path.exists():
            print(f"[{domain}] already exists, skipping")
            continue

        print(f"[{domain}] loading {dcfg['dataset']} ...")
        try:
            ds = load_dataset(dcfg["dataset"], "default", split=dcfg["split"])
        except Exception:
            try:
                ds = load_dataset(dcfg["dataset"], split=dcfg["split"])
            except Exception as e:
                print(f"[{domain}] FAILED: {e}")
                continue

        written = 0
        with open(out_path, "w") as f:
            for row in tqdm(ds, desc=domain):
                inp = row.get(dcfg["input_col"], "")
                out = row.get(dcfg["output_col"], "")
                if not inp or not out:
                    continue
                record = {
                    "domain": domain,
                    "prompt": inp.strip(),
                    "response": out.strip(),
                    "text": format_sample(inp, out)
                }
                f.write(json.dumps(record) + "\n")
                written += 1

        if written == 0:
            out_path.unlink()
            print(
                f"[{domain}] FAILED: 0 records written — check that "
                f"input_col='{dcfg['input_col']}' / output_col='{dcfg['output_col']}' "
                f"match the actual columns of '{dcfg['dataset']}' "
                f"(available columns: {list(ds.column_names)})"
            )
            continue

        print(f"[{domain}] saved {written} records -> data/processed/{domain}.jsonl")

if __name__ == "__main__":
    with open("configs/base_config.yaml") as f:
        cfg = yaml.safe_load(f)
    prepare(cfg)
