# 版本矩阵

| 场景 | 硬件 | 系统 | CANN | Python | PyTorch | torch_npu | 其他依赖 | 状态 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 环境基线 | IT22HMDA_4_S，2 芯片，128 GB HBM | Ubuntu 20.04.5 LTS，kernel 5.10.0-182，aarch64 | 9.0.0 | 3.11.4 | 2.7.1+cpu | 2.7.1.post2.dev20251226 | - | HCCL、`torch_npu` 和最小 NPU 张量已验证 |
| 推理 | IT22HMDA_4_S，2 芯片，128 GB HBM | 同上 | 9.0.0 | 3.11.4 | 2.7.1+cpu | 2.7.1.post2.dev20251226 | Transformers 5.14.1 | Qwen2.5-0.5B 单卡推理已验证 |
| 单卡 LoRA | IT22HMDA_4_S，2 芯片，128 GB HBM | 同上 | 9.2.0 | 3.11.4 | 2.7.1+cpu | 2.7.1.post2.dev20251226 | Transformers 5.14.1、PEFT 0.19.1、Datasets 5.0.0 | 5 step 训练、adapter 保存、合并和推理已验证 |
| Ascend C | IT22HMDA_4_S，2 芯片，128 GB HBM | 同上 | 待实测时确认 | 3.11.4 | 2.7.1+cpu | 2.7.1.post2.dev20251226 | - | 待验证 |

驱动基线为 25.5.5，固件查询返回 `NA`。第二章实验前服务器上的 CANN 已更新，因此推理和 LoRA 记录使用了不同的 CANN 版本。
