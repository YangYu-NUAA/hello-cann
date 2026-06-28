# 01. 模型推理与服务化

本章先跑通一个 Qwen 小模型的文本生成，再把输入长度、输出长度、耗时、显存和环境版本记录下来。后面的 profiling、算子开发、优化案例和应用接入都会复用这份 baseline。

第一版实验以单卡为主。多卡部署、HCCL 通信等内容会在参考材料里轻量介绍，不作为本章必须完成的实操。

## 学习顺序

| 文档 | 内容 |
|:---|:---|
| [transformers-torch-npu.md](transformers-torch-npu.md) | 使用 Transformers 和 `torch_npu` 跑通一次文本生成 |
| [benchmark.md](benchmark.md) | 记录一次推理实验的输入、输出和性能数据 |
| [vllm-ascend.md](vllm-ascend.md) | 把模型放到服务化推理框架里 |
| [mindie.md](mindie.md) | MindIE 相关内容 |

## 本章闭环

```text
检查环境 -> 加载模型 -> 运行生成 -> 保存结果 -> 记录 baseline -> 服务化接口
```

配套代码放在：

```text
cases/qwen/scripts/run_transformers_torch_npu.py
```

实验结果默认写入：

```text
cases/qwen/results/
```

后续章节会继续使用这份结果做 profiling 和优化前后对比。模型服务跑稳后，也会作为 `hello-agent`、`hello-claw` 等应用项目的后端模型。
