# 00. 环境配置

本章检查昇腾 NPU 和 CANN 开发环境，并运行一个最小 `torch_npu` 程序。

## 本章内容

1. [环境检查](environment-checklist.md)：查看 NPU、驱动、CANN 和 Python 环境。
2. [`torch_npu` 验证](torch-npu-check.md)：运行最小 NPU 张量计算。
3. [Docker 使用](docker.md)：使用容器时需要了解的设备挂载方式。
4. [章节 Notebook](../../../notebooks/00-environment-check.ipynb)：在 Jupyter 中执行同样的检查。

## 完成标准

- `npu-smi info` 能列出可用 NPU。
- 已加载 CANN 环境，或当前镜像已经配置好相关环境变量。
- `torch_npu` 可以导入。
- `torch.npu.is_available()` 返回 `True`。
- 最小张量计算输出正确结果。

完成后继续阅读 [01. 模型推理](../01-inference/README.md)。
