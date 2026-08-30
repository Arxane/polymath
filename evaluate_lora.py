import argparse
import csv
import json
import math
import os
import re
import time

from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

try:
    import sympy as sp
except ImportError:
    raise ImportError(
        "SymPy is required. Install it with:\n"
        "pip install sympy"
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_json_or_jsonl(path):
    """
    Load either JSON or JSONL test data.

    JSON format:
        [
            {
                "question": "...",
                "answer": "..."
            }
        ]

    JSONL format:
        {"question": "...", "answer": "..."}
        {"question": "...", "answer": "..."}
    """

    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:

        if path.suffix.lower() == ".jsonl":
            data = [
                json.loads(line)
                for line in f
                if line.strip()
            ]

        else:
            data = json.load(f)

    if isinstance(data, dict):

        if "data" in data:
            data = data["data"]

        elif "examples" in data:
            data = data["examples"]

    if not isinstance(data, list):
        raise ValueError(
            "Test file must contain a list of examples."
        )

    return data


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize superficial formatting differences.

    Examples:

        25π       -> 25pi
        25 * pi   -> 25*pi
        3x²       -> 3x^2
        x = 5     -> x=5
        $68       -> 68
    """

    s = str(text).strip().lower()

    replacements = {
        "×": "*",
        "·": "*",
        "−": "-",
        "–": "-",
        "—": "-",
        "π": "pi",
        "²": "^2",
        "³": "^3",
        "⁴": "^4",
        "⁵": "^5",
    }

    for old, new in replacements.items():
        s = s.replace(old, new)

    # Remove dollar/currency symbols
    s = s.replace("$", "")
    s = s.replace("€", "")
    s = s.replace("£", "")

    # Remove LaTeX delimiters
    s = s.replace("\\(", "")
    s = s.replace("\\)", "")
    s = s.replace("\\[", "")
    s = s.replace("\\]", "")

    # \boxed{answer}
    s = re.sub(
        r"\\boxed\s*\{([^{}]*)\}",
        r"\1",
        s
    )

    # \text{...}
    s = re.sub(
        r"\\text\s*\{([^{}]*)\}",
        r"\1",
        s
    )

    # \mathrm{...}
    s = re.sub(
        r"\\mathrm\s*\{([^{}]*)\}",
        r"\1",
        s
    )

    # Remove LaTeX spacing
    s = s.replace("\\,", "")
    s = s.replace("\\!", "")
    s = s.replace("\\;", "")

    # Simple \frac{a}{b}
    while re.search(
        r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
        s
    ):
        s = re.sub(
            r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
            r"(\1)/(\2)",
            s
        )

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Normalize operators
    s = re.sub(
        r"\s*([=+\-*/^(),])\s*",
        r"\1",
        s
    )

    return s.strip(" .")


# ============================================================
# ANSWER EXTRACTION
# ============================================================

def extract_final_answer(text):
    """
    Extract the most likely final answer.

    Priority:

        1. #### answer
        2. \\boxed{answer}
        3. Final answer: ...
        4. The answer is ...
        5. Therefore ...
        6. Equation near the end
        7. Last useful line

    This is intentionally heuristic because LLM outputs are
    not guaranteed to follow one exact format.
    """

    text = str(text).strip()

    if not text:
        return ""

    # --------------------------------------------------------
    # 1. GSM8K-style:
    #
    # #### 72
    # --------------------------------------------------------

    matches = re.findall(
        r"####\s*(.+?)(?:\n|$)",
        text,
        flags=re.IGNORECASE
    )

    if matches:
        return matches[-1].strip()

    # --------------------------------------------------------
    # 2. \boxed{...}
    # --------------------------------------------------------

    matches = re.findall(
        r"\\boxed\s*\{([^{}]*)\}",
        text,
        flags=re.IGNORECASE
    )

    if matches:
        return matches[-1].strip()

    # --------------------------------------------------------
    # 3. Explicit final answer
    # --------------------------------------------------------

    patterns = [

        r"final\s+answer\s*(?:is|:)\s*(.+?)(?:\n|$)",

        r"the\s+final\s+answer\s*(?:is|:)\s*(.+?)(?:\n|$)",

        r"the\s+answer\s*(?:is|:)\s*(.+?)(?:\n|$)",

        r"answer\s*(?:is|:)\s*(.+?)(?:\n|$)",

        r"the\s+result\s*(?:is|:)\s*(.+?)(?:\n|$)",
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if matches:
            return matches[-1].strip()

    # --------------------------------------------------------
    # 4. "Therefore ..."
    # --------------------------------------------------------

    matches = re.findall(
        r"therefore[,:]?\s*(?:the\s+answer\s+is\s*)?(.+?)(?:\n|$)",
        text,
        flags=re.IGNORECASE
    )

    if matches:
        return matches[-1].strip()

    # --------------------------------------------------------
    # 5. Look for final equation
    #
    # Example:
    #   So, 17 × 24 = 408.
    #
    # We capture the RHS.
    # --------------------------------------------------------

    equation_matches = re.findall(
        r"=\s*([^\n.]+)",
        text
    )

    if equation_matches:

        candidate = equation_matches[-1].strip()

        # Don't return something obviously non-answer-like.
        if candidate:
            return candidate

    # --------------------------------------------------------
    # 6. Last useful line
    # --------------------------------------------------------

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if lines:
        return lines[-1]

    return text


# ============================================================
# NUMERIC PARSING
# ============================================================

def parse_numeric(value):
    """
    Convert simple numerical expressions into floats.

    Handles:

        408
        408.0
        1/2
        0.5
        -4%
        $68
        68 cm
        60 km/h
    """

    s = normalize_text(value)

    # Remove common units
    s = re.sub(
        r"\s*(?:"
        r"cm\^?2|cm²|"
        r"km/h|"
        r"km|"
        r"kg|"
        r"g|"
        r"liters?|litres?|"
        r"hours?|hrs?|"
        r"minutes?|mins?|"
        r"square\s+centimeters?"
        r")\s*$",
        "",
        s,
        flags=re.IGNORECASE
    )

    # Percentage
    percentage = "%" in s
    s = s.replace("%", "")

    # Fraction
    fraction = re.fullmatch(
        r"([-+]?\d+(?:\.\d+)?)"
        r"\s*/\s*"
        r"([-+]?\d+(?:\.\d+)?)",
        s
    )

    if fraction:

        denominator = float(fraction.group(2))

        if denominator == 0:
            return None

        value = (
            float(fraction.group(1))
            / denominator
        )

    else:

        try:
            value = float(s)

        except ValueError:
            return None

    if percentage:
        value /= 100.0

    return value


# ============================================================
# SYMBOLIC EQUIVALENCE
# ============================================================

def symbolic_equivalent(a, b):
    """
    Check whether two mathematical expressions are symbolically
    equivalent using SymPy.

    Examples:

        3*x^2 == 3*x**2
        x^2 + 2*x == x*(x+2)
        25*pi == pi*25
    """

    try:

        a = normalize_text(a)
        b = normalize_text(b)

        # Convert caret exponentiation
        a = a.replace("^", "**")
        b = b.replace("^", "**")

        # Handle common mathematical notation
        a = a.replace("pi", "pi")
        b = b.replace("pi", "pi")

        expr_a = sp.sympify(
            a,
            locals={
                "pi": sp.pi,
                "e": sp.E,
            }
        )

        expr_b = sp.sympify(
            b,
            locals={
                "pi": sp.pi,
                "e": sp.E,
            }
        )

        difference = sp.simplify(
            expr_a - expr_b
        )

        return difference == 0

    except Exception:
        return False


# ============================================================
# EXTRACT NUMBERS
# ============================================================

def extract_numbers(text):
    """
    Extract numerical values from text.

    Example:

        "x = 3 and x = 4"

    -> ["3", "4"]
    """

    return re.findall(
        r"[-+]?\d+(?:\.\d+)?",
        str(text)
    )


# ============================================================
# ANSWER MATCHING
# ============================================================

def answers_match(prediction, expected):
    """
    Determine whether prediction and expected answer are
    equivalent.

    Matching order:

        1. Exact normalized string
        2. Numeric equivalence
        3. x = value vs value
        4. Multiple numerical answers
        5. Symbolic equivalence
    """

    pred_raw = extract_final_answer(prediction)
    exp_raw = str(expected)

    pred = normalize_text(pred_raw)
    exp = normalize_text(exp_raw)

    # --------------------------------------------------------
    # 1. Exact normalized match
    # --------------------------------------------------------

    if pred == exp:
        return True

    # --------------------------------------------------------
    # 2. Numeric equivalence
    # --------------------------------------------------------

    pred_num = parse_numeric(pred)
    exp_num = parse_numeric(exp)

    if pred_num is not None and exp_num is not None:

        return math.isclose(
            pred_num,
            exp_num,
            rel_tol=1e-8,
            abs_tol=1e-8
        )

    # --------------------------------------------------------
    # 3. x = 5 vs 5
    # --------------------------------------------------------

    pred_equation = re.fullmatch(
        r"[a-z]\s*=\s*(.+)",
        pred
    )

    exp_equation = re.fullmatch(
        r"[a-z]\s*=\s*(.+)",
        exp
    )

    if pred_equation and not exp_equation:

        return answers_match(
            pred_equation.group(1),
            exp
        )

    if exp_equation and not pred_equation:

        return answers_match(
            pred,
            exp_equation.group(1)
        )

    # --------------------------------------------------------
    # 4. Multiple answers
    #
    # x = 3 and x = 4
    # x = 3, 4
    # 3 and 4
    # --------------------------------------------------------

    pred_numbers = extract_numbers(pred)
    exp_numbers = extract_numbers(exp)

    if pred_numbers and exp_numbers:

        try:

            pred_values = sorted(
                float(x)
                for x in pred_numbers
            )

            exp_values = sorted(
                float(x)
                for x in exp_numbers
            )

            if len(pred_values) == len(exp_values):

                if all(
                    math.isclose(
                        a,
                        b,
                        rel_tol=1e-8,
                        abs_tol=1e-8
                    )
                    for a, b in zip(
                        pred_values,
                        exp_values
                    )
                ):
                    return True

        except ValueError:
            pass

    # --------------------------------------------------------
    # 5. Symbolic equivalence
    # --------------------------------------------------------

    if symbolic_equivalent(pred, exp):
        return True

    # --------------------------------------------------------
    # 6. Symbolic x=value vs value
    # --------------------------------------------------------

    if pred_equation:

        if symbolic_equivalent(
            pred_equation.group(1),
            exp
        ):
            return True

    if exp_equation:

        if symbolic_equivalent(
            pred,
            exp_equation.group(1)
        ):
            return True

    return False


# ============================================================
# PROMPT
# ============================================================

def build_prompt(example):
    """
    IMPORTANT:

    This exactly matches the format used during your LoRA
    training data:

        ### Instruction:
        question
        ### Response:
    """

    return (
        "### Instruction:\n"
        f"{example['question']}\n"
        "### Response:\n"
    )


# ============================================================
# GENERATION
# ============================================================

def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=128
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )

    # For device_map="auto", model.device points to the
    # appropriate execution device for this model.
    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        output = model.generate(

            **inputs,

            max_new_tokens=max_new_tokens,

            do_sample=False,

            num_beams=1,

            eos_token_id=tokenizer.eos_token_id,

            pad_token_id=tokenizer.pad_token_id,

        )

    # Only decode newly generated tokens.
    generated_tokens = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return generated_text.strip()


# ============================================================
# MODEL LOADING
# ============================================================

def get_model_kwargs(dtype):

    kwargs = {}

    if dtype == "float16":

        kwargs["dtype"] = torch.float16

    elif dtype == "bfloat16":

        kwargs["dtype"] = torch.bfloat16

    elif dtype == "float32":

        kwargs["dtype"] = torch.float32

    if torch.cuda.is_available():

        kwargs["device_map"] = "auto"

    return kwargs


def load_base(base_name, dtype):

    kwargs = get_model_kwargs(dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        base_name,
        use_fast=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        **kwargs
    )

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    return model, tokenizer


def load_lora(
    base_name,
    adapter_path,
    dtype
):

    kwargs = get_model_kwargs(dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        base_name,
        use_fast=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        **kwargs
    )

    model = PeftModel.from_pretrained(
        model,
        adapter_path
    )

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    return model, tokenizer


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    tokenizer,
    examples,
    label,
    max_new_tokens
):

    results = []

    start = time.time()

    for idx, example in enumerate(
        tqdm(
            examples,
            desc=f"Evaluating {label}"
        )
    ):

        prompt = build_prompt(example)

        output = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens
        )

        expected = str(
            example["answer"]
        )

        extracted = extract_final_answer(
            output
        )

        correct = answers_match(
            output,
            expected
        )

        results.append({

            "id": example.get(
                "id",
                idx
            ),

            "category": example.get(
                "category",
                "uncategorized"
            ),

            "difficulty": example.get(
                "difficulty",
                "unknown"
            ),

            "question": example[
                "question"
            ],

            "expected": expected,

            f"{label}_output": output,

            f"{label}_final_answer": extracted,

            f"{label}_correct": bool(
                correct
            ),

        })

    elapsed = time.time() - start

    return results, elapsed


# ============================================================
# SUMMARY
# ============================================================

def summarize(
    results,
    label
):

    total = len(results)

    correct = sum(
        bool(r[f"{label}_correct"])
        for r in results
    )

    by_category = defaultdict(
        lambda: [0, 0]
    )

    by_difficulty = defaultdict(
        lambda: [0, 0]
    )

    for r in results:

        category = r[
            "category"
        ]

        difficulty = r[
            "difficulty"
        ]

        by_category[
            category
        ][1] += 1

        by_category[
            category
        ][0] += int(
            r[f"{label}_correct"]
        )

        by_difficulty[
            difficulty
        ][1] += 1

        by_difficulty[
            difficulty
        ][0] += int(
            r[f"{label}_correct"]
        )

    return {

        "accuracy": (
            correct / total
            if total
            else 0.0
        ),

        "correct": correct,

        "total": total,

        "by_category": {

            key: {

                "accuracy": (
                    value[0] / value[1]
                    if value[1]
                    else 0.0
                ),

                "correct": value[0],

                "total": value[1],

            }

            for key, value
            in sorted(
                by_category.items()
            )

        },

        "by_difficulty": {

            key: {

                "accuracy": (
                    value[0] / value[1]
                    if value[1]
                    else 0.0
                ),

                "correct": value[0],

                "total": value[1],

            }

            for key, value
            in sorted(
                by_difficulty.items()
            )

        },

    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        required=True
    )

    parser.add_argument(
        "--adapter",
        required=True
    )

    parser.add_argument(
        "--test_file",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        default="results/math_eval"
    )

    parser.add_argument(
        "--dtype",
        choices=[
            "auto",
            "float16",
            "bfloat16",
            "float32"
        ],
        default="auto"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load test set
    # --------------------------------------------------------

    examples = load_json_or_jsonl(
        args.test_file
    )

    if not examples:

        raise ValueError(
            "No examples found in test file."
        )

    # --------------------------------------------------------
    # Determine dtype
    # --------------------------------------------------------

    dtype = args.dtype

    if dtype == "auto":

        if (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ):

            dtype = "bfloat16"

        elif torch.cuda.is_available():

            dtype = "float16"

        else:

            dtype = "float32"

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "POLYMATH — PHASE 1 MATH LoRA EVALUATION"
    )
    print("=" * 70)

    print(
        f"Base model : {args.base_model}"
    )

    print(
        f"Adapter    : {args.adapter}"
    )

    print(
        f"Test file  : {args.test_file}"
    )

    print(
        f"Samples    : {len(examples)}"
    )

    print(
        f"Device     : "
        f"{'CUDA' if torch.cuda.is_available() else 'CPU'}"
    )

    print(
        f"Dtype      : {dtype}"
    )

    print(
        f"Max tokens : {args.max_new_tokens}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # BASE MODEL
    # --------------------------------------------------------

    print()
    print(
        "[1/2] Loading and evaluating BASE model..."
    )

    base_model, tokenizer = load_base(
        args.base_model,
        dtype
    )

    base_results, base_time = evaluate_model(
        base_model,
        tokenizer,
        examples,
        "base",
        args.max_new_tokens
    )

    base_summary = summarize(
        base_results,
        "base"
    )

    del base_model

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # LORA MODEL
    # --------------------------------------------------------

    print()
    print(
        "[2/2] Loading and evaluating MATH LoRA..."
    )

    lora_model, tokenizer = load_lora(
        args.base_model,
        args.adapter,
        dtype
    )

    lora_results, lora_time = evaluate_model(
        lora_model,
        tokenizer,
        examples,
        "lora",
        args.max_new_tokens
    )

    lora_summary = summarize(
        lora_results,
        "lora"
    )

    # --------------------------------------------------------
    # COMBINE RESULTS
    # --------------------------------------------------------

    combined = []

    for base, lora in zip(
        base_results,
        lora_results
    ):

        combined.append({

            "id": base["id"],

            "category": base[
                "category"
            ],

            "difficulty": base[
                "difficulty"
            ],

            "question": base[
                "question"
            ],

            "expected": base[
                "expected"
            ],

            "base_output": base[
                "base_output"
            ],

            "base_final_answer": base[
                "base_final_answer"
            ],

            "base_correct": base[
                "base_correct"
            ],

            "lora_output": lora[
                "lora_output"
            ],

            "lora_final_answer": lora[
                "lora_final_answer"
            ],

            "lora_correct": lora[
                "lora_correct"
            ],

        })

    # --------------------------------------------------------
    # CATEGORY COMPARISON
    # --------------------------------------------------------

    categories = sorted(
        {
            example.get(
                "category",
                "uncategorized"
            )
            for example in examples
        }
    )

    category_comparison = {}

    for category in categories:

        rows = [
            row
            for row in combined
            if row["category"] == category
        ]

        base_correct = sum(
            row["base_correct"]
            for row in rows
        )

        lora_correct = sum(
            row["lora_correct"]
            for row in rows
        )

        n = len(rows)

        base_accuracy = (
            base_correct / n
            if n
            else 0.0
        )

        lora_accuracy = (
            lora_correct / n
            if n
            else 0.0
        )

        category_comparison[
            category
        ] = {

            "base_accuracy":
                base_accuracy,

            "lora_accuracy":
                lora_accuracy,

            "delta":
                lora_accuracy
                - base_accuracy,

            "n": n,

        }

    # --------------------------------------------------------
    # OVERALL DELTA
    # --------------------------------------------------------

    base_accuracy = float(
        base_summary["accuracy"]
    )

    lora_accuracy = float(
        lora_summary["accuracy"]
    )

    delta = (
        lora_accuracy
        - base_accuracy
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {

        "model":
            args.base_model,

        "adapter":
            args.adapter,

        "test_file":
            args.test_file,

        "num_samples":
            len(examples),

        "device":
            (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            ),

        "dtype":
            dtype,

        "max_new_tokens":
            args.max_new_tokens,

        "base":
            base_summary,

        "lora":
            lora_summary,

        "overall_accuracy_delta":
            delta,

        "base_evaluation_seconds":
            base_time,

        "lora_evaluation_seconds":
            lora_time,

        "category_comparison":
            category_comparison,

    }

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    predictions_path = os.path.join(
        args.output_dir,
        "predictions.json"
    )

    with open(
        predictions_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            combined,
            f,
            indent=2,
            ensure_ascii=False
        )

    summary_path = os.path.join(
        args.output_dir,
        "summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        args.output_dir,
        "predictions.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        if combined:

            writer = csv.DictWriter(
                f,
                fieldnames=combined[0].keys()
            )

            writer.writeheader()

            writer.writerows(
                combined
            )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"{'Metric':<25}"
        f"{'Base':>12}"
        f"{'Math LoRA':>14}"
        f"{'Delta':>12}"
    )

    print("-" * 70)

    print(
        f"{'Overall accuracy':<25}"
        f"{base_accuracy * 100:>11.2f}%"
        f"{lora_accuracy * 100:>13.2f}%"
        f"{delta * 100:>11.2f}%"
    )

    print("-" * 70)

    for category in categories:

        result = category_comparison[
            category
        ]

        print(
            f"{category:<25}"
            f"{result['base_accuracy'] * 100:>11.2f}%"
            f"{result['lora_accuracy'] * 100:>13.2f}%"
            f"{result['delta'] * 100:>11.2f}%"
        )

    print("=" * 70)

    print(
        f"Base correct : "
        f"{base_summary['correct']}/"
        f"{base_summary['total']}"
    )

    print(
        f"LoRA correct : "
        f"{lora_summary['correct']}/"
        f"{lora_summary['total']}"
    )

    print()

    print(
        f"Predictions: {predictions_path}"
    )

    print(
        f"CSV:         {csv_path}"
    )

    print(
        f"Summary:     {summary_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()