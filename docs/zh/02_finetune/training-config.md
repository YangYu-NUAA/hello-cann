# 单卡训练参数

## batch 与梯度累积

本章只有一个训练进程，等效 batch size 为：

```text
per_device_train_batch_size * gradient_accumulation_steps
```

最小训练使用 `1 * 1`。完整训练配置使用 `4 * 4`，等效 batch size 为 16。

## epoch 与 step

`num_train_epochs` 按数据轮数控制训练。`max_steps` 为正数时会覆盖 epoch，适合快速检查：

```bash
--max-steps 5
```

正式训练应记录样本数、epoch、global step 和 batch 配置，缺少其中一项就无法复现实验。

## 序列长度

`max_length` 决定每条样本保留的最大 token 数。长度越大，显存和计算量越高。课程样例较短，最小训练使用 128；完整配置默认 1024。

## gradient checkpointing

gradient checkpointing 会在反向传播时重新计算部分中间结果，以计算时间换显存。运行检查可用 `--no-gradient-checkpointing` 减少变量；长序列或较大 batch 时再开启。

## 保存与日志

| 参数 | 说明 |
|:---|:---|
| `logging_steps` | 每隔多少 step 输出一次训练指标 |
| `save_steps` | 每隔多少 step 保存 checkpoint |
| `output_dir` | adapter 和 tokenizer 的保存目录 |
| `report_to` | Trainer 日志后端，本章默认 `none` |

SwanLab 是可选项。没有安装或没有账号时，不影响本章训练。
