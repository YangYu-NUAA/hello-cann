# 02. 微调与训练

本章使用 Qwen2.5-0.5B-Instruct 完成单卡 LoRA 微调。训练脚本保存 adapter、训练指标和环境版本，随后将 adapter 合并回基座模型并运行一次推理。

多卡训练和 HCCL 不在本章安排实操。

## 章节内容

| 文档 | 内容 | 状态 |
|:---|:---|:---|
| [dataset-format.md](dataset-format.md) | 数据格式、chat template 和 label mask | 已完成运行检查 |
| [peft-principle.md](peft-principle.md) | LoRA 参数和 Qwen 目标模块 | 已完成 |
| [training-config.md](training-config.md) | 单卡训练参数 | 已完成 |
| [lora-single-card.md](lora-single-card.md) | 训练、保存和结果读取 | 已实测 |
| [weight-merge.md](weight-merge.md) | adapter 合并与推理验证 | 已实测 |
| [training-log.md](training-log.md) | 训练记录字段 | 已完成 |

Notebook 入口为 `notebooks/02_finetune.ipynb`。

## 已完成实验

2026 年 7 月 25 日，单卡 LoRA 最小训练在 IT22HMDA_4_S、CANN 9.2.0 环境通过。训练运行 5 step，adapter 正常保存并合并，合并后的模型可以在 `npu:0` 上生成文本。

这次实验用于检查训练、保存、加载、合并和推理，不评价微调效果。完整记录见 [lora-sft-it22hmda.md](../../../cases/qwen/reports/lora-sft-it22hmda.md)。
