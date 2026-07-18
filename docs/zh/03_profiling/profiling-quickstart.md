# profiling 快速开始

## 本节目标

对一次真实推理或训练过程采集 profile，并找到主要耗时位置。

第一轮只分析 01 章的 Transformers + `torch_npu` 推理 baseline。先把同一条命令反复跑通，再采集 profile。

## 前置条件

- 01 章推理脚本已经跑通。
- `cases/qwen/results/` 里已有一份推理 JSON。
- 本次 profile 使用同一个模型、prompt、输出长度和 dtype。

## baseline 命令

先记录一条不带 profiler 的命令：

```bash
python cases/qwen/scripts/run_transformers_torch_npu.py \
  --model /data/models/Qwen2.5-0.5B-Instruct \
  --prompt "请用三句话介绍昇腾 CANN。" \
  --max-new-tokens 128 \
  --warmup 1 \
  --repeat 3
```

把 JSON 结果路径记下来。后面的 profile 结论都要和这次 baseline 对齐。

## PyTorch profiler

先用 PyTorch profiler 做一版轻量记录。新建一个临时脚本或在 notebook 中包住同一段生成代码，记录 CPU、NPU 活动和 shape。

建议第一轮只跑一次 warmup + 一次正式生成，避免 profile 文件过大。

记录内容：

| 项目 | 说明 |
|:---|:---|
| 模型 | 与 baseline 一致 |
| prompt | 与 baseline 一致 |
| max_new_tokens | 与 baseline 一致 |
| profiler 输出目录 | 例如 `assets/03_profiling/qwen-transformers/` |
| 关键截图 | timeline 或算子表 |

## msProf / CANN profiling

如果服务器已经安装 msProf，继续采集 CANN 侧数据。不同 CANN 版本命令可能有差异，本节先记录实际可用命令，不强行统一。

需要保留：

- 采集命令。
- 输出目录。
- 关键截图或表格。
- 采集时的模型、prompt、输出长度。

## 热点表

第一轮不要只贴截图，至少整理一张表：

| 排名 | 名称 | 类型 | 耗时或占比 | shape / dtype | 备注 |
|:---|:---|:---|:---|:---|:---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |

备注里写清楚这个热点适不适合进入 04 章：

- 适合：形状固定、逻辑清楚、能写 PyTorch 参考实现。
- 暂不适合：框架调度、模型加载、数据准备、服务等待。

## 结论写法

每次 profile 只回答三个问题：

1. 时间主要花在哪里？
2. 这个判断来自哪张表或哪张图？
3. 下一步准备用什么实验验证？

没有优化前后对比时，不写“提升”。只写 baseline 和观察结果。
