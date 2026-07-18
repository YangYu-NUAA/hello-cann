# 单卡 LoRA 微调

## 本节目标

基于 Qwen 系列模型完成一次单卡 LoRA 微调，保存训练日志、权重和推理验证结果。

第一轮实验先做 smoke test：确认数据能读、模型能训、checkpoint 能保存、LoRA 权重能合并。不要一开始追求完整效果。

## 测试环境

| 项目 | 版本或说明 |
|:---|:---|
| 硬件 | IT22HMDA_4_S（2 芯片），64 GB HBM/芯片 |
| 系统 | Ubuntu 20.04.5 LTS，aarch64 |
| CANN | 9.0.0 |
| Python | 3.11.4 |
| PyTorch | 2.7.1+cpu |
| torch_npu | 2.7.1.post2.dev20251226 |
| 微调工具 | transformers + peft + datasets |
| 模型 | Qwen2.5-0.5B-Instruct 或本地同级别 Qwen 模型 |

实际版本以服务器输出为准，跑完后同步到 `docs/zh/06_reference/version-matrix.md`。

## 前置条件

先完成 00 和 01：

- `torch_npu` 可以导入。
- `torch.npu.is_available()` 返回 `True`。
- Qwen 推理 baseline 已经跑通。
- 模型权重已经下载到本地，或服务器可以访问模型仓库。

安装本节依赖：

```bash
pip install peft datasets
```

如果要接 SwanLab，再安装：

```bash
pip install swanlab
```

## 数据

第一轮使用仓库里的小样例：

```text
cases/qwen/datasets/huanhuan-100.json
```

数据是 Alpaca 三字段格式：

```json
{
  "instruction": "你是谁？",
  "input": "",
  "output": "臣妾甄嬛，参见皇上。"
}
```

这份数据只用于验证训练流程。正式实验可以替换为自己的领域数据，但字段格式先保持一致。

## 运行

回到仓库根目录：

```bash
cd /path/to/hello-cann
```

先用配置文件跑一轮：

```bash
python cases/qwen/scripts/run_lora_sft.py \
  --config cases/qwen/configs/lora-sft.example.json
```

如果模型路径不同，直接覆盖：

```bash
python cases/qwen/scripts/run_lora_sft.py \
  --config cases/qwen/configs/lora-sft.example.json \
  --model /data/models/Qwen2.5-0.5B-Instruct
```

第一次调通时可以把训练做小：

```bash
python cases/qwen/scripts/run_lora_sft.py \
  --model /data/models/Qwen2.5-0.5B-Instruct \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-length 512
```

## 输出

脚本会保存：

```text
cases/qwen/results/lora/
cases/qwen/results/lora_sft_<timestamp>.json
```

终端里至少要看到：

```text
=== LoRA SFT Done ===
train_loss: ...
global_step: ...
output_dir: ...
record_file: ...
```

JSON 结果里会记录训练参数、loss、样本数、耗时、峰值显存和版本信息。

## 合并权重

训练结束后，合并 LoRA 权重：

```bash
python cases/qwen/scripts/merge_lora.py \
  --base-model /data/models/Qwen2.5-0.5B-Instruct \
  --adapter-path cases/qwen/results/lora \
  --output-dir cases/qwen/results/lora_merged \
  --verify-prompt "你是谁？"
```

如果显存紧张，可以先跳过验证：

```bash
python cases/qwen/scripts/merge_lora.py \
  --base-model /data/models/Qwen2.5-0.5B-Instruct \
  --adapter-path cases/qwen/results/lora \
  --output-dir cases/qwen/results/lora_merged \
  --no-verify
```

## 记录

把本次结果填到：

```text
cases/qwen/reports/lora-sft-template.md
```

第一轮至少记录：

| 项目 | 说明 |
|:---|:---|
| 训练命令 | 完整命令 |
| 模型 | 本地路径或模型名 |
| 数据 | 数据文件路径和样本数 |
| batch / max_length | 实际使用值 |
| loss | JSON 里的 `train_loss` |
| 耗时 | JSON 里的 `train_total_seconds` |
| 峰值显存 | JSON 里的 `peak_memory` |
| checkpoint | LoRA 输出目录 |

## 常见问题

### `torch.npu is not available`

先回到环境章，确认 CANN 环境变量、`libhccl.so` 动态链接检查和 `torch_npu` 最小张量校验。

### 显存不足

先减小 `per_device_train_batch_size` 和 `max_length`，再考虑减少 LoRA 目标模块。第一轮只要求流程跑通。

### 训练很慢

先记录真实耗时，不要改太多参数。后面 profiling 章再分析慢在哪里。
