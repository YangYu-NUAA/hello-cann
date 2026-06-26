# 01. Inference：模型推理与服务化

这一章从最小文本生成开始，逐步进入服务化推理和 benchmark。

## 内容

| 文档 | 内容 |
|:---|:---|
| [transformers-torch-npu.md](transformers-torch-npu.md) | Transformers + `torch_npu` 最小推理 |
| [vllm-ascend.md](vllm-ascend.md) | vLLM-Ascend 服务化部署 |
| [mindie.md](mindie.md) | MindIE 部署占位 |
| [benchmark.md](benchmark.md) | 推理指标记录模板 |

## 记录指标

- 模型名称和版本。
- 输入长度、输出长度、batch、并发数。
- 首 token 延迟、吞吐、总耗时。
- 显存占用。
- CANN、PyTorch、`torch_npu`、推理框架版本。

