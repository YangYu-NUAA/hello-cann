# FAQ

## 环境

### `import torch_npu` 失败怎么办？

先检查 Python 环境、PyTorch 和 `torch_npu` 版本是否匹配，再确认 CANN 环境变量是否加载。

## 推理

### 模型加载慢怎么办？

记录模型大小、权重路径、磁盘类型、首次加载时间和二次加载时间，再判断是 I/O、图编译还是框架初始化问题。

## 训练

### 多卡训练内容在哪里？

第一阶段主线实操按单卡环境设计。多卡训练、tensor parallel、HCCL 启动方式和常见报错放在扩展说明中。

## Ascend C

### 算子正确但性能不好怎么办？

先补正确性测试、输入 shape、dtype、tiling 策略和 profile 结果，再讨论优化。

