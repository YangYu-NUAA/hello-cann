# hello-cann 中文课程

本目录是 hello-cann 的中文课程正文。

## 章节

| 章节 | 内容 |
|:---|:---|
| [00 Environment](00_environment/index.md) | 环境与版本基线 |
| [01 Inference](01_inference/index.md) | 模型推理与服务化 |
| [02 Fine-tune](02_finetune/index.md) | 单卡 LoRA 微调 |
| [03 Profiling](03_profiling/index.md) | 性能分析与瓶颈定位 |
| [04 Ascend C](04_ascendc/index.md) | Ascend C、自定义算子和 LLM 常用算子 |
| [05 Cases](05_cases/index.md) | Qwen 贯穿案例和应用项目接入 |
| [06 Reference](06_reference/index.md) | FAQ、术语和组件索引 |

## 主线案例

第一阶段围绕 Qwen 系列模型展开：

1. 检查昇腾环境和 `torch_npu`。
2. 跑通 Transformers 推理。
3. 启动推理服务并记录指标。
4. 完成单卡 LoRA 微调。
5. 用 profiling 定位热点。
6. 写一个 Ascend C 基础算子。
7. 实现一个 LLM 常用算子的 CANN 版本。
8. 接入模型链路并做正确性和性能对比。
9. 把昇腾模型服务接入一个上层应用，记录端到端任务耗时。
