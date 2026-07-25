# 01. 模型推理

本章使用 Transformers 和 `torch_npu` 在单张昇腾 NPU 上运行 Qwen，并记录生成速度与显存占用。

## 本章内容

1. [Transformers 单卡推理](transformers-torch-npu.md)：加载模型并完成文本生成。
2. [推理指标记录](benchmark.md)：记录输入长度、输出长度、延迟、吞吐和显存。
3. [vLLM-Ascend](vllm-ascend.md)：部署 OpenAI 兼容服务。
4. [MindIE](mindie.md)：了解 MindIE 的模型服务方式。
5. [章节 Notebook](../../../notebooks/01-qwen-inference.ipynb)：逐步运行 Transformers 推理。

建议先完成 Transformers 推理。vLLM-Ascend 和 MindIE 可根据使用环境选择其一。

代码位于 [`cases/qwen/scripts/`](../../../cases/qwen/scripts/)，实测数据位于 [`cases/qwen/reports/`](../../../cases/qwen/reports/)。

完成后继续阅读 [02. 单卡微调](../02-fine-tune/README.md)。
