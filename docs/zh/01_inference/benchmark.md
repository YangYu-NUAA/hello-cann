# 推理 benchmark 记录模板

推理记录的重点不是把数字写得很好看，而是让下一次实验能复现。模型、输入长度、输出长度、batch、并发、版本只要有一个不同，结果就不能直接比较。

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
python cases/qwen/scripts/run_transformers_torch_npu.py \
  --model /data/models/Qwen2.5-0.5B-Instruct \
  --prompt "请用三句话介绍昇腾 CANN。" \
  --max-new-tokens 128 \
  --warmup 1 \
  --repeat 3
```

输出目录：

```text
cases/qwen/results/
```

报告模板：

```text
cases/qwen/reports/inference-baseline-template.md
```

## 备注

- 不同模型、输入长度、并发数之间不要直接比较。
- 每次记录都要带上环境版本。
- 第一版不统计首 token 延迟。等服务化推理章节补齐流式输出后，再把 TTFT 加入记录。
