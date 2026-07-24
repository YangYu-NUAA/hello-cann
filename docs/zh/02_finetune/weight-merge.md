# 权重合并与推理验证

## 1. 记录基座模型输出

使用与合并后模型相同的 prompt：

```bash
python cases/qwen/scripts/run_transformers_torch_npu.py --model "$MODEL_PATH" --local-files-only --prompt "你是谁？" --max-new-tokens 32 --warmup 0 --repeat 1 --output-dir cases/qwen/results/lora-smoke-eval
```

## 2. 合并 adapter

```bash
python cases/qwen/scripts/merge_lora.py --base-model "$MODEL_PATH" --local-files-only --adapter-path cases/qwen/results/lora-smoke --output-dir cases/qwen/results/lora-smoke-merged --verify-prompt "你是谁？" --max-new-tokens 32
```

脚本依次完成：

1. 加载基座模型和 adapter。
2. 调用 `merge_and_unload()` 合并权重。
3. 保存模型与 tokenizer。
4. 将合并模型放到 `npu:0` 并生成文本。

## 3. 检查结果

本次 5 step 实验中，基座模型和合并模型都能正常回答“你是谁？”，回答内容接近，没有出现明显的角色风格变化。

这说明 adapter 加载、合并和 NPU 推理已经通过，但不能据此判断训练效果。效果比较应使用更充分的训练和独立验证问题。

合并模型是一份完整权重，文件明显大于 adapter。确认实验记录已经保存后，可按平台存储策略清理 `lora-smoke-merged/`。
