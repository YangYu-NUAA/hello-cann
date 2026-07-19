# Qwen2.5-0.5B 单卡推理记录

## 环境

| 项目 | 内容 |
|:---|:---|
| 日期 | 2026-07-19 |
| 机器 | IT22HMDA_4_S |
| NPU | npu:0，单芯片 64 GB HBM |
| CANN | 9.0.0 |
| Python | 3.11.4 |
| PyTorch | 2.7.1+cpu |
| torch_npu | 2.7.1.post2.dev20251226 |
| Transformers | 5.14.1 |
| 模型 | Qwen2.5-0.5B-Instruct |

## 运行命令

```bash
export MODEL_PATH=/mnt/workspace/data/Qwen2.5-0.5B-Instruct && python cases/qwen/scripts/run_transformers_torch_npu.py --model "$MODEL_PATH" --local-files-only --prompt "请用三句话介绍昇腾 CANN。" --max-new-tokens 32 --warmup 0 --repeat 1
```

## 结果

| 项目 | 数值 |
|:---|:---|
| prompt tokens | 39 |
| generated tokens | 32 |
| 平均耗时 | 1.02937 秒 |
| 输出吞吐 | 31.087 tokens/s |
| 当前分配显存 | 950.2 MB |
| 峰值分配显存 | 971.52 MB |
| 当前保留显存 | 1032.0 MB |
| 峰值保留显存 | 1032.0 MB |

这次使用 `warmup=0`、`repeat=1`，用于检查模型加载、NPU 生成和 JSON 落盘。正式 benchmark 需要固定输入，增加预热和重复次数后重新记录。

模型完成了 32 tokens 生成，但回答中存在事实错误，因此不在课程正文中引用生成内容。

## 遇到的问题

1. 最初运行的目录不是 Git 仓库，实际克隆目录为 `/mnt/workspace/hello-cann-repo`。
2. 旧脚本不支持 `--local-files-only`，切换到课程实验分支后解决。
3. Transformers 5.14.1 的 `apply_chat_template` 返回 `BatchEncoding`，脚本已兼容该返回类型。
4. `libop_plugin_atb.so` 出现文件所有者警告，但没有影响模型加载和生成。
