# 微调数据集

本目录存放 Qwen LoRA 微调用的数据集。

## huanhuan-100.json

100 条甄嬛角色扮演对话，Alpaca 格式：

```json
{
  "instruction": "皇上驾到！",
  "input": "",
  "output": "皇上万福金安。"
}
```

数据来源：[datawhalechina/hello-rocm](https://github.com/datawhalechina/hello-rocm/blob/master/src/fine-tune/datasets/huanhuan-100.json)，原始数据来自 [huanhuan-chat](https://github.com/KMnO4-zx/huanhuan-chat)。

hello-cann 使用这 100 条数据讲解 SFT 编码、LoRA 训练、验证集评估和回答对比。自己的数据可以沿用相同的三字段格式。

## 数据格式说明

字段含义：

| 字段 | 含义 | 是否必填 |
|:---|:---|:---|
| `instruction` | 用户指令 | 是 |
| `input` | 指令的补充输入，没有则置空 | 否 |
| `output` | 期望的模型回复 | 是 |

训练时脚本会用模型的 chat template 包装 `instruction` + `input`，并对这部分做 `-100` mask，只在 `output` 上计算 loss。详见 [docs/zh/02-fine-tune/dataset-format.md](../../../docs/zh/02-fine-tune/dataset-format.md)。

## 准备自己的数据

训练脚本支持 JSON 数组、JSONL，以及外层带 `data` 字段的 JSON。`instruction` 和 `output` 必须有内容，`input` 可以为空。

示例配置使用 `eval_ratio=0.1` 和 `seed=42`，将 100 条数据固定划分为 90 条训练数据和 10 条验证数据。长文本数据应先统计 token 长度，再设置 `max_length`。
