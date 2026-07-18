# 版本矩阵

| 场景 | 硬件 | 系统 | 驱动/固件 | CANN | Python | PyTorch | torch_npu | 备注 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 环境基线 | IT22HMDA_4_S，2 芯片，128 GB HBM | Ubuntu 20.04.5 LTS，kernel 5.10.0-182，aarch64 | 驱动 25.5.5，固件返回 `NA` | 9.0.0 | 3.11.4 | 2.7.1+cpu | 2.7.1.post2.dev20251226 | `torch_npu`、HCCL 和最小 NPU 张量已验证；Transformers 未安装 |
| 推理 | IT22HMDA_4_S，2 芯片，128 GB HBM | Ubuntu 20.04.5 LTS，kernel 5.10.0-182，aarch64 | 驱动 25.5.5，固件返回 `NA` | 9.0.0 | 3.11.4 | 2.7.1+cpu | 2.7.1.post2.dev20251226 | 待安装 Transformers 并完成 Qwen baseline |
| 单卡 LoRA | IT22HMDA_4_S，2 芯片，128 GB HBM | Ubuntu 20.04.5 LTS，kernel 5.10.0-182，aarch64 | 驱动 25.5.5，固件返回 `NA` | 9.0.0 | 3.11.4 | 2.7.1+cpu | 2.7.1.post2.dev20251226 | 待验证 |
| Ascend C | IT22HMDA_4_S，2 芯片，128 GB HBM | Ubuntu 20.04.5 LTS，kernel 5.10.0-182，aarch64 | 驱动 25.5.5，固件返回 `NA` | 9.0.0 | 3.11.4 | 2.7.1+cpu | 2.7.1.post2.dev20251226 | 待验证 |
