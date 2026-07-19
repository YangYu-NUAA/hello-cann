# 01. 模型推理与服务化

本章使用 Qwen2.5-0.5B-Instruct 完成单卡推理，记录模型版本、输入输出、耗时和显存。03 章会复用同一条命令采集 profile。

多卡部署和 HCCL 不在本章安排实操。

## 章节内容

| 文档 | 内容 | 状态 |
|:---|:---|:---|
| [transformers-torch-npu.md](transformers-torch-npu.md) | 使用 Transformers 和 `torch_npu` 运行 Qwen | 已实测 |
| [benchmark.md](benchmark.md) | 记录推理延迟、吞吐和显存 | 待补标准测试 |
| [vllm-ascend.md](vllm-ascend.md) | 部署 OpenAI 兼容服务 | 待实测 |
| [mindie.md](mindie.md) | 使用 MindIE 部署模型 | 待确认版本和获取方式 |

代码和配置放在 `cases/qwen/`。脚本生成的 JSON 默认写入 `cases/qwen/results/`，整理后的实验记录放在 `cases/qwen/reports/`。

## 已完成实验

2026 年 7 月 19 日，Transformers 推理在 IT22HMDA_4_S、CANN 9.0.0 环境通过。使用 Qwen2.5-0.5B-Instruct 生成 32 tokens，单次耗时 1.02937 秒，峰值分配显存 971.52 MB。

这组数据来自 `warmup=0`、`repeat=1` 的运行检查，只说明脚本可以运行，不作为正式性能结论。完整记录见 [inference-baseline-it22hmda.md](../../../cases/qwen/reports/inference-baseline-it22hmda.md)。
