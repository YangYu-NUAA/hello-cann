#!/usr/bin/env python3
"""Merge a LoRA adapter into the base model and save the merged checkpoint.

Single-card NPU only. Verifies the merged model with one generation before exit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import torch_npu  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Cannot import torch_npu. Please install the torch_npu version that "
        "matches your CANN and PyTorch environment."
    ) from exc

try:
    from peft import PeftModel
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Cannot import peft. Run: pip install peft") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model on Ascend NPU.")
    parser.add_argument("--base-model", required=True, help="Base model id or local path.")
    parser.add_argument("--adapter-path", required=True, help="LoRA adapter directory from run_lora_sft.py.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the merged model.")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model files from local paths/cache only.",
    )
    parser.add_argument(
        "--verify-prompt",
        default="你是谁？",
        help="Prompt used for the post-merge sanity generation.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-verify", action="store_true", help="Skip the sanity generation.")
    return parser.parse_args()


def resolve_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def main() -> None:
    args = parse_args()
    if not args.device.startswith("npu"):
        raise SystemExit("This script is written for Ascend NPU. Use --device npu:0.")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("torch.npu is not available. Check CANN environment and torch_npu.")
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        use_fast=False,
        local_files_only=args.local_files_only,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=resolve_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    )

    print(f"[merge] loading adapter from {args.adapter_path}")
    model = PeftModel.from_pretrained(base, args.adapter_path)
    print("[merge] merging weights ...")
    model = model.merge_and_unload()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[merge] saving merged model to {output_dir}")
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    if args.no_verify:
        print("[merge] --no-verify set, skipping sanity generation.")
        return

    model.to(device)
    model.eval()
    messages = [{"role": "user", "content": args.verify_prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print("=== Sanity Generation ===")
    print(f"prompt: {args.verify_prompt}")
    print(f"output: {text.strip()}")


if __name__ == "__main__":
    main()
