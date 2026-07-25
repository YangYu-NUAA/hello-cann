# 算子工程模板

本页记录 Ascend C 算子工程的推荐目录结构。

```text
operator-name/
├── README.md
├── scripts/
├── src/
├── tests/
└── results/
```

## 每个算子需要记录

- 输入输出 shape 和 dtype。
- tiling 策略。
- 正确性测试。
- 性能记录。
- 已知限制。

