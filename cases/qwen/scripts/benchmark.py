#!/usr/bin/env python3
"""Benchmark Qwen text generation on Ascend NPU.

Records latency, throughput and peak memory across multiple prompt lengths,
concurrency levels and repeats. Output is a JSON record that 03_profiling and
05_cases can directly consume as the optimization baseline.

Uses JSON config plus CLI overrides, writes a JSON result record, and records
NPU memory stats and package versions. Single-card only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import platform
import statistics
import time
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

# Fixed prompt set for reproducible comparison. Short / medium / long.
DEFAULT_PROMPTS = [
    "请用三句话介绍昇腾 CANN。",
    "用 Python 写一个快速排序，并解释时间复杂度。",
    "请把以下这段话改写成更正式的商务口吻：咱们这个产品挺好的，买的人挺多，用起来也方便，售后也不用操心。",
]


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", help="Optional JSON config file. CLI overrides config.")
    pre_args, remaining = pre_parser.parse_known_args()
    defaults = {
        "model": None,
        "device": "npu:0",
        "dtype": "bfloat16",
        "max_new_tokens": 128,
        "warmup": 1,
        "repeat": 3,
        "concurrency": [1],
        "prompts": list(DEFAULT_PROMPTS),
        "trust_remote_code": False,
        "do_sample": False,
        "output_dir": str(DEFAULT_OUTPUT_DIR),
    }
    defaults.update(load_config(pre_args.config))

    parser = argparse.ArgumentParser(
        parents=[pre_parser], description="Benchmark Qwen generation on Ascend NPU."
    )
    parser.set_defaults(**defaults)
    parser.add_argument("--model", default=None, help="Model id or local path.")
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default=defaults["dtype"])
    parser.add_argument("--max-new-tokens", type=int, default=defaults["max_new_tokens"])
    parser.add_argument("--warmup", type=int, default=defaults["warmup"])
    parser.add_argument("--repeat", type=int, default=defaults["repeat"])
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=defaults["concurrency"],
        help="List of concurrency levels to sweep, e.g. 1 4 8.",
    )
    parser.add_argument("--prompts", nargs="+", default=defaults["prompts"])
    parser.add_argument("--trust-remote-code", action="store_true", default=defaults["trust_remote_code"])
    parser.add_argument("--do-sample", action="store_true", default=defaults["do_sample"])
    parser.add_argument("--output-dir", default=defaults["output_dir"])
    args = parser.parse_args(remaining)
    args.config = pre_args.config
    if not args.model:
        parser.error("--model is required unless it is provided by --config")
    return args


def resolve_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def ensure_npu_available(device: str) -> torch.device:
    if not device.startswith("npu"):
        raise SystemExit("This script is written for Ascend NPU. Use --device npu:0.")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("torch.npu is not available. Check CANN environment and torch_npu.")
    return torch.device(device)


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
    for name in ("memory_allocated", "max_memory_allocated", "memory_reserved", "max_memory_reserved"):
        func = getattr(torch.npu, name, None)
        if func is None:
            continue
        try:
            memory[f"{name}_mb"] = round(float(func()) / 1024 / 1024, 2)
        except Exception:
            pass
    return memory


def encode(tokenizer: Any, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        inputs = {"input_ids": input_ids}
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def generate_once(
    model: Any,
    tokenizer: Any,
    inputs: dict[str, Any],
    max_new_tokens: int,
    do_sample: bool,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> tuple[int, float]:
    kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
        kwargs["pad_token_id"] = pad_token_id if pad_token_id is not None else eos_token_id
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id

    input_length = int(inputs["input_ids"].shape[-1])
    synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, **kwargs)
    synchronize()
    latency = time.perf_counter() - start
    generated = out[0][input_length:]
    return int(generated.numel()), latency


def bench_single(
    model: Any,
    tokenizer: Any,
    inputs_list: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    rows: list[dict[str, Any]] = []
    for idx, inputs in enumerate(inputs_list):
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        # warmup
        for _ in range(args.warmup):
            generate_once(model, tokenizer, inputs, args.max_new_tokens, args.do_sample, eos, pad)
        latencies: list[float] = []
        token_counts: list[int] = []
        for _ in range(args.repeat):
            n, lat = generate_once(model, tokenizer, inputs, args.max_new_tokens, args.do_sample, eos, pad)
            latencies.append(lat)
            token_counts.append(n)
        avg_lat = statistics.mean(latencies)
        avg_tok = statistics.mean(token_counts)
        rows.append({
            "prompt_index": idx,
            "prompt": args.prompts[idx] if idx < len(args.prompts) else f"prompt_{idx}",
            "prompt_tokens": prompt_tokens,
            "generated_tokens": int(statistics.median(token_counts)) if token_counts else 0,
            "latency_seconds": [round(x, 6) for x in latencies],
            "avg_latency_seconds": round(avg_lat, 6),
            "tokens_per_second": round(avg_tok / avg_lat, 4) if avg_lat else 0.0,
        })
    return rows


def bench_concurrency(
    model: Any,
    tokenizer: Any,
    inputs_list: list[dict[str, Any]],
    concurrency: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Concurrent generation with a thread pool.

    NPU kernels are launched on a single device. Treat this as host-side
    request concurrency for the local script, not as a service benchmark.
    """
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    # Round-robin prompts across workers so total requests = concurrency * repeat * len(prompts)
    total_requests = args.repeat * len(inputs_list)
    work: list[int] = [i % len(inputs_list) for i in range(total_requests)]

    reset_peak_memory()
    synchronize()
    wall_start = time.perf_counter()
    per_request: list[float] = []
    per_tokens: list[int] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(generate_once, model, tokenizer, inputs_list[i], args.max_new_tokens, args.do_sample, eos, pad)
            for i in work
        ]
        for fut in concurrent.futures.as_completed(futures):
            n, lat = fut.result()
            per_request.append(lat)
            per_tokens.append(n)
    wall = time.perf_counter() - wall_start
    peak_mem = read_npu_memory()

    total_tokens = sum(per_tokens)
    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "wall_seconds": round(wall, 6),
        "total_tokens": total_tokens,
        "requests_per_second": round(total_requests / wall, 4) if wall else 0.0,
        "tokens_per_second": round(total_tokens / wall, 4) if wall else 0.0,
        "avg_request_latency_seconds": round(statistics.mean(per_request), 6) if per_request else 0.0,
        "max_request_latency_seconds": round(max(per_request), 6) if per_request else 0.0,
        "peak_memory": peak_mem,
    }


def main() -> None:
    args = parse_args()
    device = ensure_npu_available(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=resolve_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    inputs_list = [encode(tokenizer, p, device) for p in args.prompts]

    # Sequential baseline (per prompt)
    reset_peak_memory()
    single_rows = bench_single(model, tokenizer, inputs_list, args)
    single_peak = read_npu_memory()

    # Concurrency sweep
    concurrency_results: list[dict[str, Any]] = []
    for c in args.concurrency:
        if c < 1:
            continue
        concurrency_results.append(bench_concurrency(model, tokenizer, inputs_list, c, args))

    record = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "config": args.config,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "sequential": {
            "rows": single_rows,
            "peak_memory": single_peak,
            "avg_tokens_per_second": round(
                statistics.mean([r["tokens_per_second"] for r in single_rows]), 4
            ) if single_rows else 0.0,
        },
        "concurrency_sweep": concurrency_results,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", "unknown"),
            "transformers": transformers.__version__,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_file = output_dir / f"benchmark_{timestamp}.json"
    record_file.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== Benchmark Done ===")
    print(f"sequential avg tokens/s: {record['sequential']['avg_tokens_per_second']}")
    print(f"concurrency levels: {[r['concurrency'] for r in concurrency_results]}")
    print(f"record_file: {record_file}")


if __name__ == "__main__":
    main()
