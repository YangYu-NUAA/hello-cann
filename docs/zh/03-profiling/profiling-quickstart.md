# Profiling 快速开始

本节对 01 章的 Transformers 推理采集 profile，并整理耗时较高的算子。

## 1. 保存 baseline

使用与 01 章相同的模型、prompt、输出长度和 dtype：

```bash
export MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
python cases/qwen/scripts/run_transformers_torch_npu.py --model "$MODEL_PATH" --local-files-only --prompt "请用三句话介绍昇腾 CANN。" --max-new-tokens 128 --warmup 1 --repeat 3
```

保存脚本生成的 JSON。后续 profile 与该记录使用相同输入。

## 2. 采集 PyTorch Profile

PyTorch Profiler 用于观察 Python、CPU 和 NPU 算子调用。采集时保留 shape 和 memory 信息，输出目录统一放在：

```text
assets/03-profiling/qwen-transformers/
```

一次预热和一次正式生成即可得到便于阅读的时间线。

## 3. 采集 CANN Profile

安装了 `msprof` 的环境可以继续采集 CANN 侧数据：

```bash
msprof --help
```

根据当前 CANN 版本选择命令参数，并在实验记录中保存完整采集命令。

## 4. 整理热点

| 排名 | 算子 | 调用次数 | 总耗时 | 输入 shape | dtype |
|:---|:---|:---|:---|:---|:---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |

在表格后说明：

1. 时间主要花在哪些算子或阶段。
2. 结论来自哪张时间线或算子表。
3. 哪个热点适合进入 04 章做算子实验。

报告字段见 [性能报告模板](report-template.md)。
