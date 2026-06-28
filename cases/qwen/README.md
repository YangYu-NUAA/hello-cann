# Qwen 主线案例

本目录存放 Qwen 系列模型的推理、微调、profiling 和算子优化案例。第一阶段先把单卡推理 baseline 跑通。

## 目录

```text
cases/qwen/
├── README.md
├── configs/
├── scripts/
├── results/
└── reports/
```

## 当前实验

### Transformers + torch_npu

```bash
python cases/qwen/scripts/run_transformers_torch_npu.py \
  --model /data/models/Qwen2.5-0.5B-Instruct \
  --prompt "请用三句话介绍昇腾 CANN。" \
  --max-new-tokens 128 \
  --warmup 1 \
  --repeat 3
```

也可以修改配置文件后运行：

```bash
python cases/qwen/scripts/run_transformers_torch_npu.py \
  --config cases/qwen/configs/transformers-torch-npu.example.json
```

结果默认写入：

```text
cases/qwen/results/
```

实验记录可以从这个模板开始：

```text
cases/qwen/reports/inference-baseline-template.md
```

## 后续内容

1. 单卡 LoRA 微调。
2. profiling。
3. Ascend C 最小算子。
4. 模型链路接入和性能对比。
