# 数据格式

本页记录 LoRA 微调使用的数据格式、切分方式和最小样例。

## JSONL 样例

```json
{"instruction": "请介绍 CANN。", "input": "", "output": "CANN 是昇腾 AI 处理器的软件栈。"}
```

## 第一轮记录

- 使用 Alpaca 三字段格式：`instruction`、`input`、`output`。
- 先用 `cases/qwen/datasets/huanhuan-100.json` 验证流程。
- tokenizer 侧使用模型自带 chat template。
- prompt 部分在 label 中写成 `-100`，只让回答部分参与 loss 计算。
- 训练集和验证集切分在完整训练前再补；第一轮 smoke test 先使用全量小样例。
