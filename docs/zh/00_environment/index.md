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
| 驱动版本 | 25.5.1（Innerversion: V100R001C23SPC006B220） |
| 固件版本 | 不支持查询 |
| CANN 版本 | 9.1.0（`~/Ascend/cann-9.1.0/`，PATH 中已包含） |
| Python 版本 | 3.11.4 |
| PyTorch / torch_npu | torch 2.7.1+cpu / torch_npu 2.7.1.post2.dev20251226（当前缺 `libhccl.so`） |
| Transformers | 未安装 |

第一组实验先完成 `libhccl.so` 排查、Transformers 安装和最小 NPU 张量校验。

## 内容

| 文档 | 内容 |
|:---|:---|
| [environment-checklist.md](environment-checklist.md) | 环境检查清单 |
| [torch-npu-check.md](torch-npu-check.md) | PyTorch + `torch_npu` 最小校验 |
| [docker.md](docker.md) | Docker 使用 |
