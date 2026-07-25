# 单卡 LoRA 微调

本节使用 Qwen2.5-0.5B-Instruct 和 100 条样例数据运行 LoRA 训练。先运行 5 step 检查训练代码，再运行 3 epoch 并计算验证集 loss。

## 1. 准备环境

按 [00 章](../00-environment/README.md) 加载 CANN 和 Python 环境，然后检查 NPU：

```bash
python -c 'import torch, torch_npu; print(torch.__version__); print(torch_npu.__version__); print(torch.npu.is_available(), torch.npu.device_count())'
```

安装训练依赖：

```bash
python -m pip install peft datasets
```

## 2. 准备模型和数据

将 `MODEL_PATH` 设为本地模型目录：

```bash
export MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
test -f "$MODEL_PATH/config.json" && echo "model ok"
```

课程数据位于：

```text
cases/qwen/datasets/huanhuan-100.json
```

数据字段、chat template 和 label mask 见 [数据格式](dataset-format.md)。

## 3. 运行 5 step 训练

```bash
python cases/qwen/scripts/run_lora_sft.py --model "$MODEL_PATH" --local-files-only --data-file cases/qwen/datasets/huanhuan-100.json --output-dir cases/qwen/results/lora-smoke --max-steps 5 --eval-ratio 0 --per-device-train-batch-size 1 --gradient-accumulation-steps 1 --max-length 128 --logging-steps 1 --save-steps 1000 --no-gradient-checkpointing
```

`--max-steps 5` 会覆盖 epoch 数。终端应当输出可训练参数量、逐 step loss 和结果文件路径：

```text
trainable params: 4,399,104 || all params: 498,431,872 || trainable%: 0.8826
...
=== LoRA SFT Done ===
train_loss: ...
global_step: 5
output_dir: cases/qwen/results/lora-smoke
record_file: cases/qwen/results/lora_sft_<timestamp>.json
```

## 4. 运行 3 epoch 训练

示例配置位于 `cases/qwen/configs/lora-sft.example.json`：

```bash
python cases/qwen/scripts/run_lora_sft.py --config cases/qwen/configs/lora-sft.example.json --model "$MODEL_PATH" --output-dir cases/qwen/results/lora-full
```

配置文件使用以下参数：

| 参数 | 值 |
|:---|:---|
| epoch | 3 |
| train / eval | 90 / 10 |
| batch size | 4 |
| 梯度累积 | 4 |
| max length | 1024 |
| learning rate | 1e-4 |
| seed | 42 |

每个 epoch 结束后会计算 eval loss。完整参数说明见 [训练参数](training-config.md)。

## 5. 比较基座模型和 adapter

```bash
python cases/qwen/scripts/compare_lora_outputs.py --base-model "$MODEL_PATH" --local-files-only --adapter-path cases/qwen/results/lora-full --prompts-file cases/qwen/datasets/lora-eval-prompts.json --max-new-tokens 64
```

脚本使用相同的 system prompt、问题和生成参数，依次生成基座模型和 LoRA adapter 的回答。结果保存在：

```text
cases/qwen/results/lora_comparison_<timestamp>.json
```

## 6. 输出文件

adapter 目录包含：

```text
adapter_config.json
adapter_model.safetensors
tokenizer.json
tokenizer_config.json
training_args.bin
```

训练记录保存在：

```text
cases/qwen/results/lora_sft_<timestamp>.json
```

课程实测数据和输出摘要见 [Qwen2.5-0.5B 单卡 LoRA 记录](../../../cases/qwen/reports/lora-sft-it22hmda.md)。

## 7. 常见问题

### 显存不足

减小 `per_device_train_batch_size` 或 `max_length`。长序列训练可以启用 gradient checkpointing。

### loss 没有明显变化

5 step 只用于检查训练程序。查看训练效果时，使用完整训练的 train loss、eval loss 和固定问题输出。

### 找不到 `libhccl.so`

重新加载 CANN 环境后，再启动训练进程。

Notebook 入口：[02-qwen-lora.ipynb](../../../notebooks/02-qwen-lora.ipynb)。
