# 权重合并与推理验证

## 1. 直接比较基座模型和 adapter

完整训练后先运行固定问题：

```bash
python cases/qwen/scripts/compare_lora_outputs.py --base-model "$MODEL_PATH" --local-files-only --adapter-path cases/qwen/results/lora-full --prompts-file cases/qwen/datasets/lora-eval-prompts.json --max-new-tokens 64
```

该步骤不生成合并模型，适合先检查 adapter。结果保存在 `cases/qwen/results/lora_comparison_<timestamp>.json`。

## 2. 需要完整权重时合并 adapter

```bash
python cases/qwen/scripts/merge_lora.py --base-model "$MODEL_PATH" --local-files-only --adapter-path cases/qwen/results/lora-full --output-dir cases/qwen/results/lora-full-merged --verify-prompt "你是谁？" --max-new-tokens 32
```

脚本依次完成：

1. 加载基座模型和 adapter。
2. 调用 `merge_and_unload()` 合并权重。
3. 保存模型与 tokenizer。
4. 将合并模型放到 `npu:0` 并生成文本。

## 3. 检查结果

5 step 实验中，基座模型和合并模型都能正常回答“你是谁？”，回答内容接近。该结果只检查 adapter 的加载、合并和 NPU 推理。

合并模型是一份完整权重，文件明显大于 adapter。确认实验记录已经保存后，可按平台存储策略清理合并目录。
