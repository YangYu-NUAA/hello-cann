# Transformers + torch_npu 最小推理

本节用 Transformers 在昇腾 NPU 上跑一次 Qwen 文本生成。跑完以后，我们会得到三样东西：

1. 一段模型输出。
2. 一份 JSON 格式的运行记录。
3. 一条可以继续做 profiling 的推理命令。

## 1. 检查环境

推理章只做运行前确认，不在这里安装或修复 CANN。先看 NPU 是否可见：

```bash
npu-smi info
```

再找一下当前环境里有没有 `set_env.sh`。有些容器的 `/usr/local/Ascend` 下面只有 `driver`，这时不要照着固定路径 `source`，先看 CANN Toolkit 实际装在哪里：

```bash
find /usr/local/Ascend ~/Ascend -name set_env.sh 2>/dev/null
```

如果能找到脚本，先加载它，例如：

```bash
source /path/to/set_env.sh
```

如果找不到脚本，但 Python 侧已经能正常导入 `torch_npu` 并使用 NPU，可以继续本章实验，同时把这个环境状态记录到环境清单里。如果 `torch_npu` 导入失败，先回到环境章处理。

再确认 Python 侧能看到 PyTorch 和 `torch_npu`：

```bash
python - <<'PY'
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu available:", hasattr(torch, "npu") and torch.npu.is_available())
PY
```

如果这里已经报错，先回到环境章处理。推理章默认 `npu-smi`、PyTorch 和 `torch_npu` 已经通过最小检查。

## 2. 安装依赖

PyTorch 和 `torch_npu` 跟 CANN 版本强相关，不在这里重复安装。这里补齐 Transformers 推理会用到的依赖：

```bash
pip install transformers accelerate safetensors sentencepiece
```

如果课程环境后面统一了版本，这里会改成固定版本的安装命令。

## 3. 准备模型

模型可以直接写 Hugging Face 仓库名，也可以写本地路径。教学环境里更推荐先下载到本地，例如：

```text
/data/models/Qwen2.5-0.5B-Instruct
```

本节默认使用一个小模型，便于快速确认链路：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

链路稳定后，可以换成更大的同系列模型。第一次跑课程时，先不要急着换大模型，先把命令、输出和记录文件跑通。

## 4. 运行脚本

回到仓库根目录：

```bash
cd /Users/mac/Documents/AI-Projects/hello-cann
```

如果模型已经下载到本地：

```bash
python cases/qwen/scripts/run_transformers_torch_npu.py \
  --model /data/models/Qwen2.5-0.5B-Instruct \
  --prompt "请用三句话介绍昇腾 CANN。" \
  --max-new-tokens 128 \
  --warmup 1 \
  --repeat 3
```

如果当前机器可以直接访问 Hugging Face：

```bash
python cases/qwen/scripts/run_transformers_torch_npu.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "请用三句话介绍昇腾 CANN。" \
  --max-new-tokens 128 \
  --warmup 1 \
  --repeat 3
```

也可以先改配置文件：

```text
cases/qwen/configs/transformers-torch-npu.example.json
```

然后用配置文件运行：

```bash
python cases/qwen/scripts/run_transformers_torch_npu.py \
  --config cases/qwen/configs/transformers-torch-npu.example.json
```

脚本默认使用：

```text
npu:0
```

需要换卡时可以加：

```bash
--device npu:1
```

## 5. 看输出

终端里会打印模型生成结果和本次实验摘要，类似：

```text
=== Generated Text ===
...

=== Metrics ===
prompt_tokens: 24
generated_tokens: 96
avg_latency_s: 1.234
tokens_per_second: 77.80
result_file: cases/qwen/results/transformers_torch_npu_20260627_041500.json
```

JSON 文件会记录更完整的信息：

```json
{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "device": "npu:0",
  "prompt_tokens": 24,
  "generated_tokens": 96,
  "latency_seconds": [1.21, 1.25, 1.24],
  "avg_latency_seconds": 1.23,
  "tokens_per_second": 78.05,
  "versions": {
    "python": "3.10.x",
    "torch": "...",
    "torch_npu": "...",
    "transformers": "..."
  }
}
```

后续做 profiling 时，直接拿同一条命令重新运行，方便比较。

## 6. 填写实验记录

跑完以后，把结果整理到：

```text
cases/qwen/reports/inference-baseline-template.md
```

第一轮只需要记录这些内容：

| 项目 | 版本或说明 |
|:---|:---|
| 模型 | 例如 Qwen2.5-0.5B-Instruct |
| 设备 | 例如 npu:0 |
| 输入 | prompt 内容 |
| 输出 | 复制一小段即可 |
| prompt tokens | 从 JSON 里抄 |
| generated tokens | 从 JSON 里抄 |
| 平均耗时 | 从 JSON 里抄 |
| 输出吞吐 | 从 JSON 里抄 |
| 结果文件 | JSON 文件路径 |

## 7. 常见问题

### 找不到 `torch_npu`

先确认当前 Python 环境是不是课程环境。很多机器上 CANN 已经装好，但 Python 虚拟环境里还没有安装对应版本的 `torch_npu`。

### `torch.npu.is_available()` 返回 `False`

先看 `npu-smi info` 是否正常，再检查 CANN 环境变量是否已经加载。

### 模型下载很慢

把模型提前下载到本地，然后把 `--model` 改成本地路径。脚本不要求模型一定来自 Hugging Face 在线仓库。

### 首次运行耗时很长

第一次加载模型会读权重、初始化设备，还可能触发一些缓存。正式记录时使用 `--warmup 1 --repeat 3`，不要只看第一次耗时。
