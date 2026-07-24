#!/usr/bin/env python3
"""Run a minimal Qwen generation test on Ascend NPU."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

try:
    import torch_npu  # noqa: F401
except ImportError as exc:  # pragma: no cover - only triggered off Ascend envs
    raise SystemExit(
        "Cannot import torch_npu. Please install the torch_npu version that "
        "matches your CANN and PyTorch environment."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "cases" / "qwen" / "results"


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        help="Optional JSON config file. CLI arguments override config values.",
    )
    pre_args, remaining = pre_parser.parse_known_args()
    defaults = {
        "prompt": "请用三句话介绍昇腾 CANN。",
        "device": "npu:0",
        "dtype": "auto",
        "max_new_tokens": 128,
        "warmup": 1,
        "repeat": 3,
        "do_sample": False,
        "temperature": 0.7,
        "top_p": 0.8,
        "trust_remote_code": False,
        "local_files_only": False,
        "output_dir": str(DEFAULT_OUTPUT_DIR),
    }
    defaults.update(load_config(pre_args.config))

    parser = argparse.ArgumentParser(
        parents=[pre_parser],
        description="Run Transformers text generation on Ascend NPU.",
    )
    parser.set_defaults(**defaults)
    parser.add_argument(
        "--model",
        default=None,
        help="Hugging Face model id or local model path.",
    )
    parser.add_argument(
        "--prompt",
        default=defaults["prompt"],
        help="Prompt used for generation.",
    )
    parser.add_argument(
        "--device",
        default=defaults["device"],
        help="Device name, for example npu:0.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default=defaults["dtype"],
        help="Model dtype. Use auto unless the environment requires a fixed dtype.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=defaults["max_new_tokens"],
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=defaults["warmup"],
        help="Warmup runs before recording latency.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=defaults["repeat"],
        help="Recorded runs.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        default=defaults["do_sample"],
        help="Use sampling instead of greedy decoding.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=defaults["temperature"],
        help="Sampling temperature. Only used with --do-sample.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=defaults["top_p"],
        help="Nucleus sampling top-p. Only used with --do-sample.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=defaults["trust_remote_code"],
        help="Pass trust_remote_code=True to Transformers.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=defaults["local_files_only"],
        help="Load model files from local cache/path only.",
    )
    parser.add_argument(
        "--output-dir",
        default=defaults["output_dir"],
        help="Directory used to save the JSON result.",
    )
    args = parser.parse_args(remaining)
    args.config = pre_args.config
    if not args.model:
        parser.error("--model is required unless it is provided by --config")
    return args


def resolve_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def ensure_npu_available(device: str) -> torch.device:
    if not device.startswith("npu"):
        raise SystemExit("This script is written for Ascend NPU. Use --device npu:0.")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit(
            "torch.npu is not available. Check CANN environment and torch_npu."
        )
    return torch.device(device)


def build_inputs(tokenizer: Any, prompt: str, device: torch.device) -> dict[str, Any]:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(encoded, Mapping):
            inputs = dict(encoded)
        else:
            inputs = {"input_ids": encoded}
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in inputs.items()}


def synchronize() -> None:
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def reset_peak_memory() -> None:
    if hasattr(torch, "npu") and hasattr(torch.npu, "reset_peak_memory_stats"):
        torch.npu.reset_peak_memory_stats()


def read_npu_memory() -> dict[str, float]:
    memory: dict[str, float] = {}
    if not hasattr(torch, "npu"):
        return memory

    for name in (
        "memory_allocated",
        "max_memory_allocated",
        "memory_reserved",
        "max_memory_reserved",
    ):
        func = getattr(torch.npu, name, None)
        if func is None:
            continue
        try:
            memory[f"{name}_mb"] = round(float(func()) / 1024 / 1024, 2)
        except Exception:
            pass
    return memory


def generate_once(
    model: Any,
    tokenizer: Any,
    inputs: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[str, int, float]:
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id

    synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    synchronize()
    latency = time.perf_counter() - start

    input_length = int(inputs["input_ids"].shape[-1])
    generated_ids = output_ids[0][input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, int(generated_ids.numel()), latency


def main() -> None:
    args = parse_args()
    device = ensure_npu_available(args.device)
    dtype = resolve_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    inputs = build_inputs(tokenizer, args.prompt, device)
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    for _ in range(args.warmup):
        generate_once(model, tokenizer, inputs, args)

    latencies: list[float] = []
    generated_tokens: list[int] = []
    generated_text = ""
    reset_peak_memory()

    for _ in range(args.repeat):
        text, token_count, latency = generate_once(model, tokenizer, inputs, args)
        generated_text = text
        generated_tokens.append(token_count)
        latencies.append(latency)

    avg_latency = statistics.mean(latencies) if latencies else 0.0
    avg_generated_tokens = statistics.mean(generated_tokens) if generated_tokens else 0.0
    tokens_per_second = avg_generated_tokens / avg_latency if avg_latency else 0.0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"transformers_torch_npu_{timestamp}.json"

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "config": args.config,
        "prompt": args.prompt,
        "generated_text": generated_text,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens[-1] if generated_tokens else 0,
        "avg_generated_tokens": round(avg_generated_tokens, 2),
        "latency_seconds": [round(item, 6) for item in latencies],
        "avg_latency_seconds": round(avg_latency, 6),
        "tokens_per_second": round(tokens_per_second, 4),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "memory": read_npu_memory(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", "unknown"),
            "transformers": transformers.__version__,
        },
    }

    result_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== Generated Text ===")
    print(generated_text.strip())
    print()
    print("=== Metrics ===")
    print(f"prompt_tokens: {prompt_tokens}")
    print(f"generated_tokens: {result['generated_tokens']}")
    print(f"avg_latency_s: {result['avg_latency_seconds']}")
    print(f"tokens_per_second: {result['tokens_per_second']}")
    print(f"result_file: {result_file}")


if __name__ == "__main__":
    main()
