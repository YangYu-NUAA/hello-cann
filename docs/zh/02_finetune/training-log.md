# 训练记录

每次实验至少保留以下字段：

| 项目 | 来源 |
|:---|:---|
| CANN、Python、PyTorch、torch_npu | 环境检查 |
| Transformers、PEFT、Datasets | Python 包版本 |
| 模型与数据集 | 训练命令 |
| LoRA r、alpha、dropout、目标模块 | JSON 记录 |
| batch、梯度累积、序列长度 | JSON 记录 |
| epoch 或 max steps | JSON 记录 |
| train loss、global step、训练耗时 | JSON 记录 |
| 当前与峰值显存 | JSON 记录 |
| adapter 和合并模型路径 | 脚本输出 |
| 微调前后回答 | 推理输出 |

模板位于：

```text
cases/qwen/reports/lora-sft-template.md
```

本次实测记录位于：

```text
cases/qwen/reports/lora-sft-it22hmda.md
```

原始 JSON、adapter 和合并模型位于 `cases/qwen/results/`，默认不提交到 Git。
