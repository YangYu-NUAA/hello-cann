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

## 脚本更新后复验

加入 `--max-steps`、`--local-files-only` 和补充记录字段后，使用下面的命令再次运行：

```bash
python cases/qwen/scripts/run_lora_sft.py --model /mnt/workspace/data/Qwen2.5-0.5B-Instruct --local-files-only --data-file cases/qwen/datasets/huanhuan-100.json --output-dir cases/qwen/results/lora-smoke-v2 --max-steps 5 --per-device-train-batch-size 1 --gradient-accumulation-steps 1 --max-length 128 --logging-steps 1 --save-steps 1000 --no-gradient-checkpointing
```

| 项目 | 数值 |
|:---|:---|
| train loss | 4.3377410412 |
| global step | 5 |
| 训练总耗时 | 1.9902065040 秒 |
| 峰值分配显存 | 1495.48 MB |
| 结果文件 | cases/qwen/results/lora_sft_20260725_020414.json |

复验记录中包含 `max_steps=5`、可训练参数量、Datasets 版本和 `cann_home=/home/developer/Ascend/cann-9.2.0`。JSONL 两条样例读取检查也已通过。

## 3 epoch 训练

2026 年 7 月 25 日继续使用同一台机器运行完整训练：

```bash
python cases/qwen/scripts/run_lora_sft.py --config cases/qwen/configs/lora-sft.example.json --model /mnt/workspace/data/Qwen2.5-0.5B-Instruct --output-dir cases/qwen/results/lora-full
```

数据按 `seed=42` 固定划分为 90 条训练数据和 10 条验证数据。

| 参数 | 值 |
|:---|:---|
| epoch | 3 |
| batch size | 4 |
| 梯度累积 | 4 |
| max length | 1024 |
| learning rate | 1e-4 |
| gradient checkpointing | True |
| global step | 18 |

| 指标 | 数值 |
|:---|:---|
| train loss | 3.8827552795 |
| epoch 1 eval loss | 4.0161752701 |
| epoch 2 eval loss | 3.9522368908 |
| epoch 3 eval loss | 3.9445757866 |
| 最终 eval loss | 3.9445755482 |
| 训练总耗时 | 24.7237181619 秒 |
| 当前分配显存 | 1001.04 MB |
| 峰值分配显存 | 2154.29 MB |
| 当前保留显存 | 3760.0 MB |
| 峰值保留显存 | 3760.0 MB |
| adapter 路径 | cases/qwen/results/lora-full |
| 结果文件 | cases/qwen/results/lora_sft_20260725_155114.json |

## 固定问题对比

```bash
python cases/qwen/scripts/compare_lora_outputs.py --base-model /mnt/workspace/data/Qwen2.5-0.5B-Instruct --local-files-only --adapter-path cases/qwen/results/lora-full --prompts-file cases/qwen/datasets/lora-eval-prompts.json --max-new-tokens 64
```

| 问题 | 基座模型 | LoRA adapter |
|:---|:---|:---|
| 娘娘，夜深了，可要歇息？ | 以助手口吻回答 | 使用“臣妾”“皇上”等角色用语 |
| 御花园的梅花开了，娘娘可要去看看？ | 解释《甄嬛传》人物和地点 | 以角色口吻直接回答 |
| 请简要说明 LoRA 微调的作用。 | 说明低秩适配及其用途 | 正常说明微调方法，没有转为宫廷对话 |

对比记录为 `cases/qwen/results/lora_comparison_20260725_155155.json`。这组结果说明 adapter 已改变角色对话的表达方式。数据量只有 100 条，不能据此评价事实准确性或通用能力。

完整训练时，Transformers 对 `warmup_ratio` 给出弃用提示。课程配置随后改为 `warmup_steps=1`。
