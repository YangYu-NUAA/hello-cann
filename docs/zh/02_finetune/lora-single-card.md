# 单卡 LoRA 微调

本节使用 Qwen2.5-0.5B-Instruct 和 100 条样例数据运行 LoRA 训练。先执行 5 step 的运行检查，再决定是否进行完整训练。

## 1. 环境

新开终端后先查找并加载 CANN 主环境脚本：

```bash
find /usr/local/Ascend ~/Ascend -name set_env.sh 2>/dev/null
```

本次实验使用：

```bash
source /home/developer/Ascend/cann-9.2.0/set_env.sh
```

不要使用 `msmemscope`、`ascendnpu-ir` 等工具目录中的同名脚本。随后进入仓库并激活 Python 环境：

```bash
cd /mnt/workspace/hello-cann-repo && source .venv/bin/activate
```

检查 NPU：

```bash
python -c "import torch, torch_npu; print(torch.__version__); print(torch_npu.__version__); print(torch.npu.is_available(), torch.npu.device_count())"
```

本次实测版本如下：

| 项目 | 版本 |
|:---|:---|
| CANN | 9.2.0 |
| Python | 3.11.4 |
| PyTorch | 2.7.1+cpu |
| torch_npu | 2.7.1.post2.dev20251226 |
| Transformers | 5.14.1 |
| PEFT | 0.19.1 |
| Datasets | 5.0.0 |

01 章推理实验使用的是 CANN 9.0.0。两次实验的软件环境不同，性能数据不直接比较。

## 2. 安装依赖

```bash
python -m pip install peft datasets
```

安装后记录版本：

```bash
python -c "import peft, datasets; print('peft', peft.__version__); print('datasets', datasets.__version__)"
```

SwanLab 只用于可选的训练监控，本次实验没有安装：

```bash
python -m pip install swanlab
```

## 3. 准备模型和数据

设置本地模型路径：

```bash
export MODEL_PATH=/mnt/workspace/data/Qwen2.5-0.5B-Instruct && test -f "$MODEL_PATH/config.json" && echo "model ok"
```

课程样例位于：

```text
cases/qwen/datasets/huanhuan-100.json
```

数据格式和 label mask 见 [dataset-format.md](dataset-format.md)。

## 4. 运行 5 step 最小训练

```bash
python cases/qwen/scripts/run_lora_sft.py --model "$MODEL_PATH" --local-files-only --data-file cases/qwen/datasets/huanhuan-100.json --output-dir cases/qwen/results/lora-smoke --max-steps 5 --per-device-train-batch-size 1 --gradient-accumulation-steps 1 --max-length 128 --logging-steps 1 --save-steps 1000 --no-gradient-checkpointing
```

`--max-steps 5` 会覆盖 `num_train_epochs`，适合检查训练代码。`--local-files-only` 禁止脚本联网查找模型文件。

终端应当看到可训练参数量、逐 step loss 和完成信息：

```text
trainable params: 4,399,104 || all params: 498,431,872 || trainable%: 0.8826
...
=== LoRA SFT Done ===
train_loss: ...
global_step: 5
output_dir: cases/qwen/results/lora-smoke
record_file: cases/qwen/results/lora_sft_....json
```

本次运行得到：

| 项目 | 数值 |
|:---|:---|
| global step | 5 |
| train loss | 4.3443 |
| 训练总耗时 | 2.9814 秒 |
| 峰值分配显存 | 1495.48 MB |
| adapter 大小 | 约 17.6 MB |

这些数字只对应 5 step 运行检查。

## 5. 完整训练

确认最小训练成功后，可以使用配置文件：

```bash
python cases/qwen/scripts/run_lora_sft.py --config cases/qwen/configs/lora-sft.example.json --model "$MODEL_PATH"
```

配置文件默认训练 3 个 epoch，batch size 为 4，梯度累积为 4，最大长度为 1024。运行前根据显存和数据长度调整参数。

训练参数见 [training-config.md](training-config.md)。

## 6. 输出文件

adapter 目录包含：

```text
adapter_config.json
adapter_model.safetensors
chat_template.jinja
tokenizer.json
tokenizer_config.json
training_args.bin
```

训练记录写入：

```text
cases/qwen/results/lora_sft_<timestamp>.json
```

整理后的记录放在 `cases/qwen/reports/`。生成的 checkpoint、合并模型和原始 JSON 不提交到 Git。

## 7. 常见问题

### `libhccl.so` 找不到

当前终端没有加载 CANN 环境。重新查找并 `source` CANN 主环境脚本，再导入 `torch_npu`。

### `Permission mismatch` 警告

本次实验中该警告没有影响训练、保存和推理。不要直接修改平台预装文件的所有者。

### 显存不足

先减小 `per_device_train_batch_size` 和 `max_length`。仍然不足时启用 gradient checkpointing。

### loss 没有明显下降

5 step 只检查程序是否可运行。判断训练效果需要增加训练步数，并准备独立验证集。
