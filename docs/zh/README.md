# hello-cann 中文课程

## 课程目录

| 章节 | 内容 | 实践入口 |
|:---|:---|:---|
| [00 环境配置](00-environment/README.md) | CANN 环境、`torch_npu` 和最小 NPU 计算 | [Notebook](../../notebooks/00-environment-check.ipynb) |
| [01 模型推理](01-inference/README.md) | Transformers 推理、benchmark 和服务化 | [Notebook](../../notebooks/01-qwen-inference.ipynb) |
| [02 大模型微调](02-fine-tune/README.md) | SFT 数据处理、LoRA 训练、验证和权重合并 | [Notebook](../../notebooks/02-qwen-lora.ipynb) |
| [03 性能分析](03-profiling/README.md) | Profiling、热点算子和性能报告 | 正文 |
| [04 算子开发](04-ascend-c/README.md) | Ascend C、Vector Add 和模型接入 | 正文 |
| [05 综合案例](05-cases/README.md) | Qwen 优化和应用项目接入 | 正文 |
| [06 参考资料](06-references/README.md) | 版本矩阵、组件、术语和常见问题 | 索引 |

## 阅读顺序

首次学习建议按 00、01、02 的顺序完成环境检查、推理和微调，再进入性能分析与算子开发。已经具备可用 CANN 环境的读者可以从 01 章开始。

课程脚本集中在 `cases/qwen/scripts/`，Ascend C 工程位于 `src/04-ascend-c/`。每章正文都附有对应的代码入口。
