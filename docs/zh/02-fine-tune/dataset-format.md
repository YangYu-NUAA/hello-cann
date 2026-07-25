# 微调数据格式

训练脚本读取 Alpaca 三字段数据：

| 字段 | 说明 |
|:---|:---|
| `instruction` | 用户指令，必填 |
| `input` | 指令的补充内容，可以为空 |
| `output` | 期望回答，必填 |

## JSON 数组

课程样例 `cases/qwen/datasets/huanhuan-100.json` 使用 JSON 数组：

```json
[
  {
    "instruction": "皇上驾到！",
    "input": "",
    "output": "皇上万福金安。"
  }
]
```

脚本也接受外层带 `data` 字段的写法：

```json
{
  "data": [
    {
      "instruction": "皇上驾到！",
      "input": "",
      "output": "皇上万福金安。"
    }
  ]
}
```

## JSONL

JSONL 每行是一条完整记录，不使用最外层数组：

```jsonl
{"instruction":"皇上驾到！","input":"","output":"皇上万福金安。"}
{"instruction":"你是谁？","input":"","output":"臣妾甄嬛，参见皇上。"}
```

## 输入与 labels

脚本使用模型自带的 chat template 处理 system prompt 和用户输入。`instruction` 与 `input` 属于提示部分，对应 labels 全部设为 `-100`；只有 `output` 和结束符参与 loss 计算。

```text
input_ids: [system 和 user tokens] [assistant tokens] [eos]
labels:    [-100, -100, ...]   [assistant tokens] [eos]
```

序列超过 `max_length` 时会从末尾截断。如果截断后没有留下回答 tokens，这条样本不能用于训练。准备长文本数据时，应先统计 token 长度，再确定 `max_length`。

## 检查课程数据

```bash
python -c "import json; d=json.load(open('cases/qwen/datasets/huanhuan-100.json')); print(type(d).__name__, len(d)); print(d[0])"
```

本次实测读入 100 条记录。首条编码后为 43 tokens，其中 18 tokens 参与 loss。
