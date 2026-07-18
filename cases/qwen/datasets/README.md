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

hello-cann 用它做 LoRA 微调的最小验证示例。跑通后建议替换为自己的领域数据（同样三字段格式即可）。

## 数据格式说明

字段含义：

| 字段 | 含义 | 是否必填 |
|:---|:---|:---|
| `instruction` | 用户指令 | 是 |
| `input` | 指令的补充输入，没有则置空 | 否 |
| `output` | 期望的模型回复 | 是 |

训练时脚本会用模型的 chat template 包装 `instruction` + `input`，并对这部分做 `-100` mask，只在 `output` 上计算 loss。详见 [docs/zh/02_finetune/dataset-format.md](../../../docs/zh/02_finetune/dataset-format.md)。

## 准备自己的数据

把你的训练数据导出成同样的 JSON 数组即可。建议：

- 至少 100-500 条做最小验证。
- 验证集单独切出来（脚本默认全量训练，可在 `run_lora_sft.py` 里加 `eval_dataset`）。
- 长文本截断到 `max_length`（默认 1024）。
