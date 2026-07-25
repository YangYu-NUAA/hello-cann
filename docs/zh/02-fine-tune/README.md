# 02. 昇腾大模型微调

这一章使用 [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) 完成一次单卡 LoRA 微调。训练任务是让模型学习《甄嬛传》人物的对话风格，过程包括数据处理、LoRA 配置、短训练、完整训练、验证集评估、回答对比和权重合并。

完成本章后，你应该能够：

- 看懂一条 SFT 数据如何变成模型的 `input_ids` 和 `labels`；
- 为 Qwen 的注意力层和前馈网络配置 LoRA；
- 在单张昇腾 NPU 上运行训练，并读懂 loss、step 和显存记录；
- 比较基座模型与 LoRA adapter 的输出；
- 保存 adapter，并在需要时合并为完整模型。

配套 Notebook：[02-qwen-lora.ipynb](../../../notebooks/02-qwen-lora.ipynb)

## 2.1 从指令数据开始

预训练模型学习的是“根据前文预测下一个 token”。经过指令微调后，模型才逐渐熟悉 system、user 和 assistant 之间的对话方式。

有监督微调（Supervised Fine-Tuning，SFT）会同时提供问题和参考回答。例如：

```json
{
  "instruction": "将下列文本翻译成英文：",
  "input": "今天天气真好",
  "output": "Today is a nice day."
}
```

训练时，`instruction` 和 `input` 组成用户消息，`output` 是模型需要学习的回答。损失函数只计算回答部分，用户消息只负责提供上下文。

Qwen2.5-0.5B-Instruct 已经具备通用对话能力。本章用少量角色对话继续训练它，观察 LoRA 是否能改变模型的表达风格。

## 2.2 课程数据集

本章使用 [`huanhuan-100.json`](../../../cases/qwen/datasets/huanhuan-100.json)，其中有 100 条《甄嬛传》角色对话。

数据来源如下：

1. 原始数据来自 [KMnO4-zx/huanhuan-chat](https://github.com/KMnO4-zx/huanhuan-chat/blob/master/dataset/train/lora/huanhuan.json)。
2. [hello-rocm](https://github.com/datawhalechina/hello-rocm/blob/master/src/fine-tune/datasets/huanhuan-100.json) 从中整理了 100 条样例。
3. hello-cann 收录同一份 100 条子集，用于昇腾单卡训练。

选择这组数据是因为样本少、训练时间短，而且角色语言变化容易观察。它适合学习训练流程，不用于衡量模型的通用能力。

先在仓库根目录查看数据：

```bash
python -c "import json; data=json.load(open('cases/qwen/datasets/huanhuan-100.json', encoding='utf-8')); print('样本数:', len(data)); print(data[0])"
```

其中一条记录如下：

```json
{
  "instruction": "娘娘。",
  "input": "",
  "output": "你放心，本宫到任何时候都不会自轻自贱委屈了这孩子。"
}
```

三个字段的作用是：

| 字段 | 作用 |
|:---|:---|
| `instruction` | 用户提出的指令或对话上文 |
| `input` | 额外输入，没有时使用空字符串 |
| `output` | 模型需要学习的回答 |

训练脚本也接受 JSONL 和外层包含 `data` 字段的 JSON。替换成自己的数据时，保持这三个字段即可，详细格式见[微调数据格式](dataset-format.md)。

本章另有一份 [`lora-eval-prompts.json`](../../../cases/qwen/datasets/lora-eval-prompts.json)。里面的问题没有出现在训练集中，用来比较微调前后的回答，不参与训练。

## 2.3 数据如何送入 Qwen

不同模型使用的对话标记不同。课程脚本不手写 `<|im_start|>` 等特殊 token，而是调用 Qwen tokenizer 自带的 chat template：

```python
prompt_text = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ],
    tokenize=False,
    add_generation_prompt=True,
)
```

一条样本最终被整理为：

```text
system 消息 + user 消息 + assistant 起始标记 + output + eos
```

模型输入包含整段对话，但 labels 只保留回答部分：

```text
input_ids: [system 和 user tokens] [assistant tokens] [eos]
labels:    [-100, -100, ...]   [assistant tokens] [eos]
```

Transformers 在计算交叉熵时会忽略值为 `-100` 的位置。因此，模型看到问题，但只学习如何生成回答。

完整处理代码位于 [`run_lora_sft.py`](../../../cases/qwen/scripts/run_lora_sft.py) 的 `build_dataset()`。Notebook 会读取第一条样本，并打印总 token 数与参与 loss 的 token 数。

序列超过 `max_length` 时会被截断。对长文本数据，应先统计长度分布，保证截断后仍保留足够的回答 token。

## 2.4 为什么使用 LoRA

全量微调会更新模型的所有参数。LoRA 冻结基座模型，在选定的线性层旁增加两个低秩矩阵，只训练新增参数：

```text
W' = W + (alpha / r) * B * A
```

本章使用以下配置：

| 参数 | 取值 | 含义 |
|:---|:---|:---|
| `r` | 8 | 低秩矩阵的秩 |
| `lora_alpha` | 32 | LoRA 更新的缩放系数 |
| `lora_dropout` | 0.1 | LoRA 分支的 dropout |
| `task_type` | `CAUSAL_LM` | 因果语言模型任务 |

LoRA 被插入 Qwen 的七类线性层：

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

前四个属于注意力模块，后三个属于前馈网络。Qwen2.5-0.5B-Instruct 在这组配置下有 4,399,104 个可训练参数，约占模型总参数的 0.88%。

LoRA 参数的进一步说明见 [LoRA 参数](peft-principle.md)。

## 2.5 准备环境和模型

开始前应完成 [00. 环境配置](../00-environment/README.md) 和 [01. 模型推理](../01-inference/README.md)，确保 `torch_npu` 能识别 NPU，Qwen 模型可以正常生成文本。

安装本章依赖：

```bash
python -m pip install peft datasets
```

检查环境：

```bash
python -c 'import torch, torch_npu, transformers, peft, datasets; print("NPU:", torch.npu.is_available()); print("transformers:", transformers.__version__); print("peft:", peft.__version__); print("datasets:", datasets.__version__)'
```

已经下载模型时，设置模型目录：

```bash
export MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
```

也可以使用 Hugging Face CLI 下载：

```bash
python -m pip install -U huggingface_hub
```

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir "$MODEL_PATH"
```

确认目录中存在模型配置：

```bash
test -f "$MODEL_PATH/config.json" && echo "model ok"
```

如果服务器可以直接访问 Hugging Face，也可以在训练命令中把 `--model` 写成 `Qwen/Qwen2.5-0.5B-Instruct`，并去掉 `--local-files-only`。

## 2.6 先运行 5 step

第一次运行使用较短序列和较小 batch，只训练 5 step：

```bash
python cases/qwen/scripts/run_lora_sft.py --model "$MODEL_PATH" --local-files-only --data-file cases/qwen/datasets/huanhuan-100.json --output-dir cases/qwen/results/lora-smoke --max-steps 5 --eval-ratio 0 --per-device-train-batch-size 1 --gradient-accumulation-steps 1 --max-length 128 --logging-steps 1 --save-steps 1000 --no-gradient-checkpointing
```

这一步用于检查：

1. 数据能否读取和编码；
2. 模型能否在 `npu:0` 上完成反向传播；
3. LoRA 参数能否保存；
4. 训练指标能否写入 JSON。

终端会先打印可训练参数量，结束时给出 loss、step、输出目录和记录文件：

```text
trainable params: 4,399,104
...
=== LoRA SFT Done ===
train_loss: ...
global_step: 5
output_dir: cases/qwen/results/lora-smoke
record_file: cases/qwen/results/lora_sft_<timestamp>.json
```

5 step 只能说明训练程序工作正常，不能用来判断模型是否已经学会角色风格。

## 2.7 完成 3 epoch 训练

短训练通过后，使用课程配置运行 3 epoch：

```bash
python cases/qwen/scripts/run_lora_sft.py --config cases/qwen/configs/lora-sft.example.json --model "$MODEL_PATH" --output-dir cases/qwen/results/lora-full
```

配置文件的主要参数如下：

| 参数 | 取值 |
|:---|:---|
| 训练集 / 验证集 | 90 / 10 |
| epoch | 3 |
| 单卡 batch size | 4 |
| 梯度累积 | 4 |
| 等效 batch size | 16 |
| `max_length` | 1024 |
| 学习率 | 1e-4 |
| 随机种子 | 42 |

`seed=42` 固定了数据划分。每个 epoch 结束后，Trainer 会在 10 条验证数据上计算 eval loss。

课程实测时，eval loss 从 4.0161 降到 3.9446。完整环境、命令、耗时和显存见 [Qwen2.5-0.5B 单卡 LoRA 记录](../../../cases/qwen/reports/lora-sft-it22hmda.md)。

batch、梯度累积、序列长度和 gradient checkpointing 的关系见[单卡训练参数](training-config.md)。

## 2.8 比较微调前后的回答

训练 loss 只能说明优化目标在变化。角色风格是否发生改变，还要用训练集之外的问题比较输出。

```bash
python cases/qwen/scripts/compare_lora_outputs.py --base-model "$MODEL_PATH" --local-files-only --adapter-path cases/qwen/results/lora-full --prompts-file cases/qwen/datasets/lora-eval-prompts.json --max-new-tokens 64
```

脚本会使用相同的 system prompt 和生成参数，分别运行基座模型与 LoRA adapter。课程实测得到的现象是：

| 问题 | 基座模型 | LoRA adapter |
|:---|:---|:---|
| 娘娘，夜深了，可要歇息？ | 以普通助手口吻回答 | 使用“臣妾”“皇上”等角色用语 |
| 御花园的梅花开了，娘娘可要去看看？ | 解释人物和地点 | 以角色口吻直接回答 |
| 请简要说明 LoRA 微调的作用。 | 正常解释 LoRA | 仍能回答一般问题 |

对比结果保存在：

```text
cases/qwen/results/lora_comparison_<timestamp>.json
```

这组结果可以说明 adapter 改变了角色对话的表达方式。100 条数据不足以评价事实准确性，也不适合用来比较模型的通用能力。

## 2.9 保存 adapter 与合并权重

训练目录中的主要文件是：

```text
adapter_config.json
adapter_model.safetensors
tokenizer.json
tokenizer_config.json
training_args.bin
```

adapter 只保存 LoRA 参数。推理时可以加载基座模型，再加载 adapter；这样占用磁盘少，也方便为不同任务保存多份 adapter。

需要一份普通 Transformers 模型时，可以合并权重：

```bash
python cases/qwen/scripts/merge_lora.py --base-model "$MODEL_PATH" --local-files-only --adapter-path cases/qwen/results/lora-full --output-dir cases/qwen/results/lora-full-merged --verify-prompt "你是谁？" --max-new-tokens 32
```

脚本会调用 `merge_and_unload()`，保存合并模型，并在 `npu:0` 上运行一次生成检查。合并模型包含完整权重，所需磁盘空间接近基座模型。

详细说明见[权重合并与推理验证](weight-merge.md)。

## 2.10 使用 Notebook

[02-qwen-lora.ipynb](../../../notebooks/02-qwen-lora.ipynb) 与本章使用相同的脚本和数据。Notebook 默认执行数据检查和 5 step 训练，完整训练、回答对比和权重合并由三个开关控制。

在仓库根目录启动：

```bash
export MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
```

```bash
python -m jupyter lab
```

## 补充阅读

- [微调数据格式](dataset-format.md)
- [LoRA 参数](peft-principle.md)
- [单卡训练参数](training-config.md)
- [单卡 LoRA 运行步骤](lora-single-card.md)
- [训练记录字段](training-log.md)
- [权重合并与推理验证](weight-merge.md)
- [Transformers Chat templates](https://huggingface.co/docs/transformers/chat_templating)
- [PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora)

完成本章后，继续阅读 [03. 性能分析](../03-profiling/README.md)。
