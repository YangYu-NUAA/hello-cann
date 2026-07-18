# 00. Environment：环境与版本基线

这一章负责把机器变成可以继续学习 CANN 的环境。

## 目标

- 确认 NPU、驱动、固件、CANN、Python、PyTorch 和 `torch_npu` 是否匹配。
- 跑通最小 NPU 校验程序。
- 记录环境检查清单和版本矩阵。
- 收集安装、权限、环境变量、算子缺失等常见问题。

## 第一组实验环境

| 项目 | 值 |
|:---|:---|
| 昇腾卡型号 | IT22HMDA_4_S（2 芯片） |
| 单芯片显存 | 64 GB HBM（共 128 GB） |
| 操作系统 | Ubuntu 20.04.5 LTS，kernel 5.10.0-182（aarch64） |
| 驱动版本 | 25.5.5（Innerversion: V100R001C23SPC009B220） |
| 固件版本 | `npu-smi` 返回 `NA` |
| CANN 版本 | 9.0.0（`~/Ascend/cann-9.0.0/`） |
| Python 版本 | 3.11.4 |
| PyTorch / torch_npu | torch 2.7.1+cpu / torch_npu 2.7.1.post2.dev20251226 |
| Transformers | 未安装 |

这套环境已通过 `torch_npu` 导入和最小 NPU 张量校验。Transformers 在 01 Inference 章安装。

## 内容

| 文档 | 内容 |
|:---|:---|
| [environment-checklist.md](environment-checklist.md) | 环境检查清单 |
| [torch-npu-check.md](torch-npu-check.md) | PyTorch + `torch_npu` 最小校验 |
| [docker.md](docker.md) | Docker 使用 |

## 本章验收

- [x] `npu-smi info` 能看到 2 个 NPU 芯片。
- [x] 已确认驱动和 CANN 版本。
- [x] `libhccl.so` 可以被动态链接器找到。
- [x] `torch_npu` 可以导入，`torch.npu.is_available()` 返回 `True`。
- [x] 最小 NPU 张量计算输出全 2 张量。
- [x] 已记录当前环境没有 Docker CLI；这不影响后续裸机实验。
