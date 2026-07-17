# Qwen 主线案例

本目录存放 Qwen 系列模型的推理、微调、benchmark、profiling 和算子优化案例。是 hello-cann 的主线案例工程。

## 目录

```text
cases/qwen/
├── README.md
├── configs/                          # 各脚本的可复现配置
│   ├── transformers-torch-npu.example.json
│   ├── lora-sft.example.json
│   └── benchmark.example.json
├── datasets/                         # 微调数据集
│   ├── README.md
│   └── huanhuan-100.json             # 示例数据集（甄嬛角色扮演）
├── scripts/                          # 可运行脚本
│   ├── run_transformers_torch_npu.py # 01 章推理 baseline
│   ├── run_lora_sft.py               # 02 章单卡 LoRA 微调
│   ├── merge_lora.py                 # 02 章 LoRA 权重合并
│   └── benchmark.py                  # 01/03 章 benchmark
├── results/                          # 脚本输出（JSON 记录、checkpoint）
└── reports/                          # 实验记录（Markdown）
    └── inference-baseline-template.md
```

## 当前实验

### 1. Transformers + torch_npu 推理 baseline（01 章）

```bash
python cases/qwen/scripts/run_transformers_torch_npu.py \
  --model /data/models/Qwen2.5-0.5B-Instruct \
  --prompt "请用三句话介绍昇腾 CANN。" \
  --max-new-tokens 128 --warmup 1 --repeat 3
```

或用配置文件：

```bash
python cases/qwen/scripts/run_transformers_torch_npu.py \
  --config cases/qwen/configs/transformers-torch-npu.example.json
```

### 2. 单卡 LoRA 微调（02 章）

```bash
python cases/qwen/scripts/run_lora_sft.py \
  --model /data/models/Qwen2.5-0.5B-Instruct \
  --data-file cases/qwen/datasets/huanhuan-100.json \
  --num-train-epochs 3
```

或用配置文件：

```bash
python cases/qwen/scripts/run_lora_sft.py \
  --config cases/qwen/configs/lora-sft.example.json
```

合并 LoRA 权重并验证：

```bash
python cases/qwen/scripts/merge_lora.py \
  --base-model /data/models/Qwen2.5-0.5B-Instruct \
  --adapter-path cases/qwen/results/lora \
  --output-dir cases/qwen/results/lora_merged \
  --verify-prompt "你是谁？"
```

### 3. benchmark（01 / 03 章）

```bash
python cases/qwen/scripts/benchmark.py \
  --model /data/models/Qwen2.5-0.5B-Instruct \
  --concurrency 1 2 4 --repeat 3
```

或用配置文件：

```bash
python cases/qwen/scripts/benchmark.py \
  --config cases/qwen/configs/benchmark.example.json
```

结果默认写入：

```text
cases/qwen/results/
```

这里的 benchmark 用来记录本地推理基线。服务化推理的接口压测会在 vLLM-Ascend 或 MindIE 小节单独记录。

## 环境依赖

推理：

```bash
pip install transformers accelerate safetensors sentencepiece
```

微调：

```bash
pip install peft datasets
# 可选：SwanLab 训练监控
pip install swanlab
```

## 后续内容

1. profiling（复用 benchmark.py 的命令采集 profile）。
2. Ascend C 最小算子（见 `src/04_ascendc/vector_add/`）。
3. 模型链路接入和性能对比（见 `docs/zh/05_cases/qwen-optimization.md`）。

## 相关文档

- 推理教程：[docs/zh/01_inference/transformers-torch-npu.md](../../docs/zh/01_inference/transformers-torch-npu.md)
- 微调教程：[docs/zh/02_finetune/lora-single-card.md](../../docs/zh/02_finetune/lora-single-card.md)
- 主线案例：[docs/zh/05_cases/qwen-optimization.md](../../docs/zh/05_cases/qwen-optimization.md)
