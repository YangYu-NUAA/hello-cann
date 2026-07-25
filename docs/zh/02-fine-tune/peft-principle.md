# LoRA 参数

LoRA 不直接更新原始权重矩阵 \(W\)，而是训练两个较小的矩阵 \(A\) 和 \(B\)：

```text
W' = W + (alpha / r) * B * A
```

基座模型参数保持冻结，checkpoint 只保存新增矩阵，因此文件远小于完整模型。

## 主要参数

| 参数 | 作用 | 本章取值 |
|:---|:---|:---|
| `r` | 低秩矩阵的秩；增大后参数量和表达能力都会增加 | 8 |
| `alpha` | LoRA 更新的缩放系数 | 32 |
| `dropout` | LoRA 分支上的 dropout | 0.1 |
| `target_modules` | 插入 LoRA 的线性层 | Qwen 注意力和 FFN 投影层 |

本章使用以下目标模块：

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

前四个属于注意力投影，后三个属于前馈网络。Qwen2.5-0.5B-Instruct 在该配置下共有 4,399,104 个可训练参数，占总参数约 0.88%。

## adapter 与合并模型

训练结束后得到的是 adapter，使用时有两种方式：

1. 加载基座模型，再通过 PEFT 加载 adapter。
2. 使用 `merge_and_unload()` 把 LoRA 更新合并到基座权重，保存一份普通 Transformers 模型。

第一种方式节省磁盘，适合保留多个任务 adapter；第二种方式部署步骤更少，但会生成一份完整模型。
