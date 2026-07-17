# LLM 常用算子的 CANN 实现

这一页记录如何从 PyTorch 参考实现出发，写出一个可以验证的 CANN 版本。

第一版先选一个算子。重点是把路径跑通：参考实现、输入构造、正确性测试、Ascend C 实现、性能记录、profile。后续再扩展到更多 LLM kernel。

## 候选算子

| 算子 | 为什么适合教学 |
|:---|:---|
| Vector Add / Scale | 最小数据搬运和 Vector API 练习 |
| RMSNorm | LLM 中常见，能练习规约、广播和精度对比 |
| Softmax | 适合观察 shape、数值稳定性和性能差异 |
| SiLU / SwiGLU | 和前馈网络相关，后续可以接到 Qwen 案例 |

## 写法

1. 先写 PyTorch 参考实现。
2. 固定输入 shape、dtype、随机种子和误差阈值。
3. 写 Ascend C 实现和 host 侧调用。
4. 用同一组输入做正确性测试。
5. 记录耗时、profile 结果和显存占用。
6. 选一个位置尝试接回模型链路。

## 记录模板

| 项目 | 内容 |
|:---|:---|
| 算子名称 |  |
| 输入 shape |  |
| dtype |  |
| PyTorch 参考耗时 |  |
| CANN 实现耗时 |  |
| 最大误差 |  |
| 平均误差 |  |
| profile 结论 |  |
| 是否接入模型 |  |

## 参考来源

`llm-algo-leetcode` 中的 PyTorch、Triton 和 CUDA 题目可以作为素材来源。这里不展开 CUDA 或 Triton 教学，只借用题目、测试方式和性能记录方式。
