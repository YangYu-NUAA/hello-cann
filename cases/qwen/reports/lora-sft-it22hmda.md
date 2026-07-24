# Qwen2.5-0.5B 单卡 LoRA 记录

## 环境

| 项目 | 内容 |
|:---|:---|
| 日期 | 2026-07-25 |
| 机器 | IT22HMDA_4_S |
| 设备 | npu:0，单芯片 64 GB HBM |
| CANN | 9.2.0 |
| Python | 3.11.4 |
| PyTorch | 2.7.1+cpu |
| torch_npu | 2.7.1.post2.dev20251226 |
| Transformers | 5.14.1 |
| PEFT | 0.19.1 |
| Datasets | 5.0.0 |
| 基座模型 | Qwen2.5-0.5B-Instruct |
| 数据集 | huanhuan-100.json，100 条 |

## 实际运行命令

```bash
python cases/qwen/scripts/run_lora_sft.py --model /mnt/workspace/data/Qwen2.5-0.5B-Instruct --data-file cases/qwen/datasets/huanhuan-100.json --output-dir cases/qwen/results/lora-smoke --num-train-epochs 0.05 --per-device-train-batch-size 1 --gradient-accumulation-steps 1 --max-length 128 --logging-steps 1 --save-steps 1000 --no-gradient-checkpointing
```

该命令来自脚本加入 `--max-steps` 之前的首轮实验，0.05 epoch 实际运行 5 step。更新后的课程命令使用 `--max-steps 5`。

## LoRA 配置

| 参数 | 值 |
|:---|:---|
| r | 8 |
| alpha | 32 |
| dropout | 0.1 |
| target modules | q/k/v/o_proj、gate/up/down_proj |
| 可训练参数 | 4,399,104 |
| 总参数 | 498,431,872 |
| 可训练参数比例 | 0.8826% |

## 训练配置

| 参数 | 值 |
|:---|:---|
| batch size | 1 |
| 梯度累积 | 1 |
| max length | 128 |
| epoch | 0.05 |
| global step | 5 |
| learning rate | 1e-4 |
| gradient checkpointing | False |

## 训练结果

| 指标 | 数值 |
|:---|:---|
| train loss | 4.3442568302 |
| 训练总耗时 | 2.9813509050 秒 |
| 当前分配显存 | 1001.04 MB |
| 峰值分配显存 | 1495.48 MB |
| 当前保留显存 | 1806.0 MB |
| 峰值保留显存 | 1806.0 MB |
| adapter 权重 | 17,640,136 bytes |
| adapter 路径 | cases/qwen/results/lora-smoke |
| 结果文件 | cases/qwen/results/lora_sft_20260725_015333.json |

## 合并与验证

```bash
python cases/qwen/scripts/merge_lora.py --base-model /mnt/workspace/data/Qwen2.5-0.5B-Instruct --adapter-path cases/qwen/results/lora-smoke --output-dir cases/qwen/results/lora-smoke-merged --verify-prompt "你是谁？" --max-new-tokens 32
```

| 模型 | 输出摘要 |
|:---|:---|
| 基座模型 | 自述为 Qwen，并介绍用途 |
| 合并模型 | 自述为 Qwen，并介绍用途 |

两次输出接近。5 step 训练只用于验证代码，没有足够依据判断角色风格是否学成。

## 实验中遇到的问题

1. 新终端没有加载 CANN 环境时，导入 `torch_npu` 报 `libhccl.so` 找不到。
2. 服务器上的 CANN 已从 9.0.0 更新为 9.2.0，第二章按 9.2.0 记录。
3. `libop_plugin_atb.so` 文件所有者警告没有阻塞训练和推理。
4. Python 环境出现 Requests 依赖版本警告，没有阻塞本次实验，后续统一环境时再处理。
