# 推理 benchmark 记录模板

模型、输入长度、输出长度、batch、并发和软件版本一致时，测试结果才适合放在一起比较。

## 最小记录

| 项目 | 说明 |
|:---|:---|
| 模型 | 模型名称或本地路径 |
| 推理框架 | Transformers + torch_npu / vLLM-Ascend / MindIE |
| 设备 | npu:0 等 |
| 输入长度 | prompt tokens |
| 输出长度 | generated tokens |
| batch | 本章第一版固定为 1 |
| 并发 | 本章第一版固定为 1 |
| 总耗时 | 多次 repeat 的平均值 |
| 输出吞吐 | generated tokens / 总耗时 |
| 峰值显存 | 能采集就记录，采集不到写明原因 |
| 测试命令 | 保留完整命令 |
| 结果文件 | JSON 或日志路径 |

## 本章脚本

Transformers 最小推理脚本会自动生成 JSON 记录：

```bash
export MODEL_PATH=/data/models/Qwen2.5-0.5B-Instruct && python cases/qwen/scripts/run_transformers_torch_npu.py --model "$MODEL_PATH" --local-files-only --prompt "请用三句话介绍昇腾 CANN。" --max-new-tokens 128 --warmup 1 --repeat 3
```

输出目录：

```text
cases/qwen/results/
```

报告模板：

```text
cases/qwen/reports/inference-baseline-template.md
```

`cases/qwen/scripts/benchmark.py` 主要用于本地推理基线。脚本里的并发测试是主机侧同时发起多次生成请求，用来观察本地调用的变化，不等同于 vLLM-Ascend 或 MindIE 的服务化压测。服务化章节会单独记录接口延迟、吞吐和首 token 时间。

2026 年 7 月 19 日已经完成一次 `warmup=0`、`repeat=1` 的运行检查。该数据记录在 `cases/qwen/reports/inference-baseline-it22hmda.md`，不与正式 benchmark 混用。

## 备注

- 不同模型、输入长度、并发数之间不要直接比较。
- 每次记录都要带上环境版本。
- 第一版不统计首 token 延迟。等服务化推理章节补齐流式输出后，再把 TTFT 加入记录。
