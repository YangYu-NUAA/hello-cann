# Transformers + torch_npu 单卡推理

本节使用 Transformers 在昇腾 NPU 上运行 Qwen2.5-0.5B-Instruct。配套脚本会保存生成结果、耗时、吞吐、显存和软件版本。

## 1. 检查环境

```bash
npu-smi info
python -c 'import torch, torch_npu; print("torch:", torch.__version__); print("torch_npu:", torch_npu.__version__); print("NPU available:", torch.npu.is_available())'
```

如果新终端尚未加载 CANN，可以按 [00 章环境检查](../00-environment/environment-checklist.md) 设置 `CANN_SET_ENV`。

## 2. 安装推理依赖

```bash
python -m pip install transformers accelerate safetensors sentencepiece
```

查看版本：

```bash
python -c 'import transformers; print(transformers.__version__)'
```

## 3. 准备模型

脚本支持本地模型目录和 Hugging Face 模型名称。

使用本地模型时，设置 `MODEL_PATH`：

```bash
export MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
test -f "$MODEL_PATH/config.json" && echo "model ok"
```

模型目录应包含 `config.json`、tokenizer 文件和模型权重。课程仓库不保存模型权重。

## 4. 运行推理

在仓库根目录执行：

```bash
python cases/qwen/scripts/run_transformers_torch_npu.py --model "$MODEL_PATH" --local-files-only --prompt "请用三句话介绍昇腾 CANN。" --max-new-tokens 32 --warmup 0 --repeat 1
```

确认推理正常后，加入预热并增加输出长度：

```bash
python cases/qwen/scripts/run_transformers_torch_npu.py --model "$MODEL_PATH" --local-files-only --prompt "请用三句话介绍昇腾 CANN。" --max-new-tokens 128 --warmup 1 --repeat 3
```

可以访问 Hugging Face 时，直接使用模型名称：

```bash
python cases/qwen/scripts/run_transformers_torch_npu.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "请用三句话介绍昇腾 CANN。" --max-new-tokens 128 --warmup 1 --repeat 3
```

也可以从配置文件运行：

```bash
python cases/qwen/scripts/run_transformers_torch_npu.py --config cases/qwen/configs/transformers-torch-npu.example.json --model "$MODEL_PATH"
```

脚本默认使用 `npu:0`。使用其他设备时添加 `--device npu:1`。

## 5. 查看结果

终端会输出生成文本和指标：

```text
=== Generated Text ===
[模型输出]

=== Metrics ===
prompt_tokens: ...
generated_tokens: ...
avg_latency_s: ...
tokens_per_second: ...
result_file: cases/qwen/results/transformers_torch_npu_<timestamp>.json
```

JSON 文件还包含当前显存、峰值显存和软件版本。已完成的 Qwen2.5-0.5B-Instruct 实验见 [单卡推理记录](../../../cases/qwen/reports/inference-baseline-it22hmda.md)。

## 6. 参数说明

| 参数 | 说明 |
|:---|:---|
| `--model` | Hugging Face 模型名称或本地模型目录 |
| `--local-files-only` | 只读取本地文件和缓存 |
| `--device` | NPU 设备，默认 `npu:0` |
| `--dtype` | 权重类型 |
| `--max-new-tokens` | 最大生成 token 数 |
| `--warmup` | 正式计时前的预热次数 |
| `--repeat` | 计时次数 |
| `--output-dir` | JSON 结果目录 |

## 7. 常见问题

### 找不到 `libhccl.so`

当前终端没有加载完整 CANN 环境。重新加载 `CANN_SET_ENV` 后再运行 Python。

### `--local-files-only` 无法识别

查看当前脚本参数：

```bash
python cases/qwen/scripts/run_transformers_torch_npu.py -h
```

更新仓库后，帮助信息中应包含 `--local-files-only`。

### `AttributeError` 出现在 `input_ids.shape`

旧版脚本没有兼容 Transformers 返回的 `BatchEncoding`。更新仓库后重新运行。

### 本地模型仍然访问网络

确认 `MODEL_PATH` 指向完整模型目录，并使用 `--local-files-only`。

### 第一次运行耗时较长

第一次运行包含权重读取和设备初始化。记录性能时使用 `--warmup 1` 或更大的预热次数。

Notebook 入口：[01-qwen-inference.ipynb](../../../notebooks/01-qwen-inference.ipynb)。
