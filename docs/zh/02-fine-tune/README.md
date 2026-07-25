# 02. 单卡微调

本章使用 Qwen2.5-0.5B-Instruct 完成 LoRA 微调，包括数据编码、训练、验证集评估、adapter 对比和权重合并。

## 本章内容

1. [数据格式](dataset-format.md)：Alpaca 数据、chat template 和 label mask。
2. [LoRA 原理](peft-principle.md)：LoRA 参数和 Qwen 目标模块。
3. [训练参数](training-config.md)：batch、梯度累积、序列长度和验证集。
4. [单卡 LoRA 实践](lora-single-card.md)：运行训练并读取结果。
5. [训练记录](training-log.md)：保存 loss、耗时、显存和环境版本。
6. [权重合并与推理](weight-merge.md)：比较 adapter 输出并按需合并权重。
7. [章节 Notebook](../../../notebooks/02-qwen-lora.ipynb)：运行数据检查、短训练和完整训练。

训练代码和配置位于 [`cases/qwen/`](../../../cases/qwen/)，实测记录见 [Qwen2.5-0.5B 单卡 LoRA 记录](../../../cases/qwen/reports/lora-sft-it22hmda.md)。

完成后继续阅读 [03. 性能分析](../03-profiling/README.md)。
