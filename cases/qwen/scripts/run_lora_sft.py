#!/usr/bin/env python3
"""Run a single-card LoRA SFT on Ascend NPU.

Uses JSON config plus CLI overrides, writes a JSON result record, and records
NPU memory stats and package versions. Single-card only.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import datasets
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
import transformers

try:
    import torch_npu  # noqa: F401
except ImportError as exc:  # pragma: no cover - only triggered off Ascend envs
    raise SystemExit(
        "Cannot import torch_npu. Please install the torch_npu version that "
        "matches your CANN and PyTorch environment."
    ) from exc

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Cannot import peft. Run: pip install peft"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "cases" / "qwen" / "results" / "lora"
DEFAULT_DATA_FILE = REPO_ROOT / "cases" / "qwen" / "datasets" / "huanhuan-100.json"


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
        "model": None,
        "device": "npu:0",
        "dtype": "bfloat16",
        "data_file": str(DEFAULT_DATA_FILE),
        "system_prompt": "现在你要扮演皇帝身边的女人--甄嬛。",
        "max_length": 1024,
        # LoRA
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        # Training
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 3,
        "max_steps": -1,
        "learning_rate": 1e-4,
        "logging_steps": 10,
        "save_steps": 100,
        "warmup_ratio": 0.03,
        "gradient_checkpointing": True,
        "seed": 42,
        "report_to": "none",
        # SwanLab (optional)
        "swanlab_project": "hello-cann",
        "swanlab_experiment": "qwen-lora-sft",
        "use_swanlab": False,
        "trust_remote_code": False,
        "local_files_only": False,
    }
    defaults.update(load_config(pre_args.config))

    parser = argparse.ArgumentParser(
        parents=[pre_parser],
        description="Run single-card LoRA SFT on Ascend NPU.",
    )
    parser.set_defaults(**defaults)
    parser.add_argument("--model", default=None, help="Base model id or local path.")
    parser.add_argument("--device", default=defaults["device"], help="Device, e.g. npu:0.")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default=defaults["dtype"],
    )
    parser.add_argument("--data-file", default=defaults["data_file"], help="Alpaca JSON file.")
    parser.add_argument("--system-prompt", default=defaults["system_prompt"])
    parser.add_argument("--max-length", type=int, default=defaults["max_length"])
    parser.add_argument("--lora-r", type=int, default=defaults["lora_r"])
    parser.add_argument("--lora-alpha", type=int, default=defaults["lora_alpha"])
    parser.add_argument("--lora-dropout", type=float, default=defaults["lora_dropout"])
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=defaults["lora_target_modules"],
    )
    parser.add_argument("--output-dir", default=defaults["output_dir"])
    parser.add_argument("--per-device-train-batch-size", type=int, default=defaults["per_device_train_batch_size"])
    parser.add_argument("--gradient-accumulation-steps", type=int, default=defaults["gradient_accumulation_steps"])
    parser.add_argument("--num-train-epochs", type=float, default=defaults["num_train_epochs"])
    parser.add_argument(
        "--max-steps",
        type=int,
        default=defaults["max_steps"],
        help="Stop after this many optimizer steps. Positive values override epochs.",
    )
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--logging-steps", type=int, default=defaults["logging_steps"])
    parser.add_argument("--save-steps", type=int, default=defaults["save_steps"])
    parser.add_argument("--warmup-ratio", type=float, default=defaults["warmup_ratio"])
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=defaults["gradient_checkpointing"],
    )
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--report-to", default=defaults["report_to"])
    parser.add_argument("--swanlab-project", default=defaults["swanlab_project"])
    parser.add_argument("--swanlab-experiment", default=defaults["swanlab_experiment"])
    parser.add_argument("--use-swanlab", action="store_true", default=defaults["use_swanlab"])
    parser.add_argument("--trust-remote-code", action="store_true", default=defaults["trust_remote_code"])
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=defaults["local_files_only"],
        help="Load model files from local paths/cache only.",
    )
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


def load_records(data_file: str) -> list[dict[str, Any]]:
    path = Path(data_file)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Dataset is empty: {path}")

    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raw = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]

    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]
    if not isinstance(raw, list):
        raise ValueError("Dataset must be a JSON array, JSONL file, or {'data': [...]}.")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset row {index} is not an object.")
        if not str(item.get("instruction", "")).strip():
            raise ValueError(f"Dataset row {index} has no instruction.")
        if not str(item.get("output", "")).strip():
            raise ValueError(f"Dataset row {index} has no output.")
        records.append(item)
    return records


def build_dataset(
    tokenizer: Any,
    data_file: str,
    system_prompt: str,
    max_length: int,
) -> Dataset:
    """Load Alpaca-format JSON and tokenize with the model chat template.

    Labels mask instruction tokens with -100 so loss is only computed on the response.
    Keeps the prompt, response and label mask logic explicit for teaching.
    """
    ds = Dataset.from_list(load_records(data_file))

    eos_token = tokenizer.eos_token or "<|im_end|>"

    def process(example: dict[str, Any]) -> dict[str, Any]:
        instruction = str(example.get("instruction", "")).strip()
        extra_input = str(example.get("input", "")).strip()
        user = f"{instruction}\n{extra_input}" if extra_input else instruction
        response = str(example.get("output", "")).strip()
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        instruction = tokenizer(prompt_text, add_special_tokens=False)
        response_enc = tokenizer(response + eos_token, add_special_tokens=False)
        input_ids = instruction["input_ids"] + response_enc["input_ids"]
        attention_mask = instruction["attention_mask"] + response_enc["attention_mask"]
        labels = [-100] * len(instruction["input_ids"]) + response_enc["input_ids"]
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return ds.map(process, remove_columns=ds.column_names)


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


def maybe_swanlab_callback(args: argparse.Namespace):
    if not args.use_swanlab:
        return None
    try:
        from swanlab.integration.huggingface import SwanLabCallback
    except ImportError:
        print("[warn] use_swanlab=True but swanlab not installed; skipping.")
        return None
    return SwanLabCallback(
        project=args.swanlab_project,
        experiment_name=args.swanlab_experiment,
    )


class MetricsTrainer(Trainer):
    """Trainer that records peak memory and total training time into trainer.state."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._train_start: float | None = None

    def train(self, *args: Any, **kwargs: Any):
        if hasattr(torch, "npu") and hasattr(torch.npu, "reset_peak_memory_stats"):
            torch.npu.reset_peak_memory_stats()
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
        self._train_start = time.perf_counter()
        result = super().train(*args, **kwargs)
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
        self.state.train_total_seconds = (
            time.perf_counter() - self._train_start if self._train_start else None
        )
        self.state.train_peak_memory = read_npu_memory()
        return result


def main() -> None:
    args = parse_args()
    device = ensure_npu_available(args.device)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        use_fast=False,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = build_dataset(tokenizer, args.data_file, args.system_prompt, args.max_length)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=resolve_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    )
    model.to(device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=args.lora_target_modules,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())

    if args.gradient_checkpointing:
        try:
            model.enable_input_require_grads()
        except Exception:
            pass

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        warmup_ratio=args.warmup_ratio,
        gradient_checkpointing=args.gradient_checkpointing,
        save_on_each_node=False,
        report_to=args.report_to,
        seed=args.seed,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    callbacks = [c for c in [maybe_swanlab_callback(args)] if c is not None]

    trainer = MetricsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    train_result = trainer.train()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    record = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "config": args.config,
        "data_file": args.data_file,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_percent": round(trainable_params / total_params * 100, 6),
        },
        "training": {
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_train_epochs": args.num_train_epochs,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
        "metrics": {
            "train_loss": (
                float(train_result.training_loss)
                if train_result.training_loss is not None
                else None
            ),
            "global_step": train_result.global_step,
            "total_samples": len(train_dataset),
            "train_total_seconds": getattr(trainer.state, "train_total_seconds", None),
            "peak_memory": getattr(trainer.state, "train_peak_memory", {}),
        },
        "output_dir": str(output_dir),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", "unknown"),
            "transformers": transformers.__version__,
            "peft": __import__("peft").__version__,
            "datasets": datasets.__version__,
            "cann_home": os.environ.get("ASCEND_HOME_PATH", ""),
        },
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_file = output_dir.parent / f"lora_sft_{timestamp}.json"
    record_file.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== LoRA SFT Done ===")
    print(f"train_loss: {record['metrics']['train_loss']}")
    print(f"global_step: {record['metrics']['global_step']}")
    print(f"output_dir: {record['output_dir']}")
    print(f"record_file: {record_file}")


if __name__ == "__main__":
    main()
