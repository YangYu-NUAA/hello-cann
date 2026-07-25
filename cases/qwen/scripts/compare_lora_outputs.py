#!/usr/bin/env python3
"""Compare base-model and LoRA-adapter outputs on Ascend NPU."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import torch_npu  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Cannot import torch_npu. Check the CANN and torch_npu environment.") from exc

try:
    import peft
    from peft import PeftModel
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Cannot import peft. Run: pip install peft") from exc


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPTS_FILE = REPO_ROOT / "cases" / "qwen" / "datasets" / "lora-eval-prompts.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "cases" / "qwen" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and LoRA outputs on Ascend NPU.")
    parser.add_argument("--base-model", required=True, help="Base model id or local path.")
    parser.add_argument("--adapter-path", required=True, help="LoRA adapter directory.")
    parser.add_argument("--prompts-file", default=str(DEFAULT_PROMPTS_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_prompts(path: str) -> tuple[str, list[dict[str, str]]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    system_prompt = str(data.get("system_prompt", "")).strip()
    prompts = data.get("prompts")
    if not system_prompt:
        raise ValueError("prompts file has no system_prompt")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts file has no prompts")
    for index, item in enumerate(prompts):
        if not isinstance(item, dict) or not str(item.get("prompt", "")).strip():
            raise ValueError(f"prompt {index} is invalid")
    return system_prompt, prompts


def generate(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    system_prompt: str,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    elapsed = time.perf_counter() - start
    generated_ids = output[0][inputs["input_ids"].shape[-1]:]
    return {
        "output": tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
        "generated_tokens": int(generated_ids.shape[-1]),
        "latency_seconds": elapsed,
    }


def main() -> None:
    args = parse_args()
    if not args.device.startswith("npu"):
        raise SystemExit("This script is written for Ascend NPU. Use --device npu:0.")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("torch.npu is not available. Check CANN environment and torch_npu.")
    device = torch.device(args.device)
    system_prompt, prompts = load_prompts(args.prompts_file)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        use_fast=False,
        local_files_only=args.local_files_only,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=resolve_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    )
    base_model.to(device)
    base_model.eval()

    comparisons = []
    for item in prompts:
        base_result = generate(
            base_model,
            tokenizer,
            device,
            system_prompt,
            item["prompt"],
            args.max_new_tokens,
        )
        comparisons.append(
            {
                "id": item.get("id", ""),
                "prompt": item["prompt"],
                "note": item.get("note", ""),
                "base": base_result,
            }
        )

    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.to(device)
    model.eval()
    for item, comparison in zip(prompts, comparisons):
        comparison["lora"] = generate(
            model,
            tokenizer,
            device,
            system_prompt,
            item["prompt"],
            args.max_new_tokens,
        )

    record = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "device": args.device,
        "dtype": args.dtype,
        "prompts_file": args.prompts_file,
        "system_prompt": system_prompt,
        "max_new_tokens": args.max_new_tokens,
        "comparisons": comparisons,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", "unknown"),
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"lora_comparison_{datetime.now():%Y%m%d_%H%M%S}.json"
    output_file.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== Base vs LoRA ===")
    for item in comparisons:
        print(f"[{item['id']}] {item['prompt']}")
        print(f"base: {item['base']['output']}")
        print(f"lora: {item['lora']['output']}")
    print(f"record_file: {output_file}")


if __name__ == "__main__":
    main()
