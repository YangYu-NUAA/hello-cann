# 04. Ascend C：自定义算子

这一章从基础算子开始，进入 tiling、数据搬运、编译运行、调试和模型接入。第一版先保证 Vector Add 和一个 LLM 常用算子能跑通、能验证、能记录结果。

## 内容

| 文档 | 内容 |
|:---|:---|
| [vector-add.md](vector-add.md) | Vector Add 入门 |
| [operator-template.md](operator-template.md) | 算子工程模板 |
| [llm-operators.md](llm-operators.md) | LLM 常用算子的 CANN 实现 |
| [model-integration.md](model-integration.md) | 接入模型链路 |
