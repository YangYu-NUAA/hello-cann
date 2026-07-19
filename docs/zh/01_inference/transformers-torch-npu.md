# Transformers + torch_npu 最小推理

本节使用 Transformers 在昇腾 NPU 上运行 Qwen2.5-0.5B-Instruct。脚本会打印生成结果，并把版本、耗时、吞吐和显存写入 JSON 文件。

## 1. 检查环境

先确认 NPU 可见：

```bash
npu-smi info
```

推理章不安装 CANN。如果当前终端没有加载 CANN 环境变量，可以先查找环境脚本：

```bash
find /usr/local/Ascend ~/Ascend -name set_env.sh 2>/dev/null
```

选择 CANN Toolkit 的主环境脚本，不要使用 `msmemscope`、`ascendnpu-ir` 等工具目录里的同名脚本。例如：

```bash
source ~/Ascend/cann-9.0.0/set_env.sh
```

部分云平台已经在终端启动时加载了环境变量。即使没有找到脚本，只要下面的检查通过，也可以继续实验：

```bash
python -c 'import torch, torch_npu; print("torch:", torch.__version__); print("torch_npu:", torch_npu.__version__); print("npu available:", torch.npu.is_available()); print("npu count:", torch.npu.device_count())'
```

如果 `torch_npu` 无法导入，或 `torch.npu.is_available()` 返回 `False`，先回到 [00.4 Python 环境与 torch_npu 验证](../00_environment/torch-npu-check.md)。

## 2. 安装依赖

PyTorch 和 `torch_npu` 应当在环境章配置完成。本节只安装模型推理依赖：

```bash
python -m pip install transformers accelerate safetensors sentencepiece
```

实验完成后记录实际安装的 Transformers 版本：

```bash
python -c 'import transformers; print(transformers.__version__)'
```

## 3. 准备模型

脚本既可以读取本地模型，也可以使用 Hugging Face 模型名。云平台可能无法稳定访问外网，建议提前下载模型，并记录保存位置。

例如，模型位于：

```text
/mnt/workspace/data/Qwen2.5-0.5B-Instruct
```

先检查模型目录：

```bash
export MODEL_PATH=/mnt/workspace/data/Qwen2.5-0.5B-Instruct && test -f "$MODEL_PATH/config.json" && echo "model ok"
```

其他机器只需要把 `MODEL_PATH` 改成自己的模型路径。课程仓库不保存模型权重。

## 4. 运行脚本

进入克隆后的仓库目录，例如：

```bash
cd /mnt/workspace/hello-cann-repo
```

使用本地模型做一次运行检查：

```bash
export MODEL_PATH=/mnt/workspace/data/Qwen2.5-0.5B-Instruct && python cases/qwen/scripts/run_transformers_torch_npu.py --model "$MODEL_PATH" --local-files-only --prompt "请用三句话介绍昇腾 CANN。" --max-new-tokens 32 --warmup 0 --repeat 1
```

`--local-files-only` 会禁止脚本联网查找模型文件。确认脚本可以运行后，再增加预热和重复次数：

```bash
export MODEL_PATH=/mnt/workspace/data/Qwen2.5-0.5B-Instruct && python cases/qwen/scripts/run_transformers_torch_npu.py --model "$MODEL_PATH" --local-files-only --prompt "请用三句话介绍昇腾 CANN。" --max-new-tokens 128 --warmup 1 --repeat 3
```

如果机器可以访问 Hugging Face，也可以直接使用模型名：

```bash
python cases/qwen/scripts/run_transformers_torch_npu.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "请用三句话介绍昇腾 CANN。" --max-new-tokens 128 --warmup 1 --repeat 3
```

配置文件入口是 `cases/qwen/configs/transformers-torch-npu.example.json`：

```bash
python cases/qwen/scripts/run_transformers_torch_npu.py --config cases/qwen/configs/transformers-torch-npu.example.json
```

脚本默认使用 `npu:0`。指定其他设备时加上 `--device npu:1`。

## 5. 查看结果

2026 年 7 月 19 日的运行检查得到以下摘要：

```text
=== Generated Text ===
[模型输出]

=== Metrics ===
prompt_tokens: 39
generated_tokens: 32
avg_latency_s: 1.02937
tokens_per_second: 31.087
result_file: cases/qwen/results/transformers_torch_npu_20260719_185039.json
```

对应环境为 Python 3.11.4、PyTorch 2.7.1+cpu、torch_npu 2.7.1.post2.dev20251226、Transformers 5.14.1。JSON 中还记录了显存数据：

```json
{
  "memory_allocated_mb": 950.2,
  "max_memory_allocated_mb": 971.52,
  "memory_reserved_mb": 1032.0,
  "max_memory_reserved_mb": 1032.0
}
```

小模型的生成内容可能包含事实错误。本节只用它确认模型加载、NPU 生成和结果落盘是否正常，不把生成文本当作 CANN 知识材料。

## 6. 整理实验记录

脚本生成的 JSON 保留原始数据。提交课程记录时，参照 `cases/qwen/reports/inference-baseline-template.md` 整理以下字段：

| 项目 | 记录内容 |
|:---|:---|
| 环境 | NPU、CANN、Python、PyTorch、torch_npu、Transformers |
| 模型 | 模型名称和本地路径 |
| 输入输出 | prompt、prompt tokens、generated tokens、输出摘要 |
| 参数 | device、dtype、warmup、repeat、max_new_tokens |
| 性能 | 平均耗时、输出吞吐、峰值显存 |
| 复现 | 完整命令和 JSON 文件路径 |

本次实测记录见 `cases/qwen/reports/inference-baseline-it22hmda.md`。

## 7. 常见问题

### `Permission mismatch` 警告

实验环境出现过下面的警告：

```text
UserWarning: Permission mismatch: The owner of .../libop_plugin_atb.so does not match.
```

当 `torch_npu` 可以导入、模型能够加载并完成生成时，这条警告不会阻塞本节实验。不要直接使用 `sudo chown` 修改平台预装的 Python 文件。若随后出现动态库加载失败，再把完整日志和文件权限交给平台管理员检查。

### `--local-files-only` 无法识别

先确认当前目录确实是 Git 仓库，并查看脚本参数：

```bash
git rev-parse --show-toplevel && python cases/qwen/scripts/run_transformers_torch_npu.py -h
```

如果帮助信息中没有 `--local-files-only`，当前脚本不是最新版本。拉取课程分支后再运行，不要只替换命令。

### `AttributeError` 出现在 `input_ids.shape`

Transformers 5.14.1 中，`apply_chat_template` 可能返回 `BatchEncoding`。旧脚本把它再次放进 `input_ids` 后会触发这个错误。课程脚本已经兼容该返回类型，更新仓库后重新运行即可。

### 找不到 `torch_npu`

确认当前使用的是课程虚拟环境，再检查 PyTorch、`torch_npu` 与 CANN 的版本组合。不要在推理章里直接升级其中一个包。

### 本地模型仍然访问网络

确认 `--model` 指向包含 `config.json` 和权重文件的目录，并加上 `--local-files-only`。

### 第一次运行较慢

第一次运行包含权重读取和设备初始化。运行检查可以使用 `--warmup 0 --repeat 1`，正式记录使用 `--warmup 1 --repeat 3`。
