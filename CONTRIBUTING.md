# 贡献指南

感谢参与 hello-cann。

## 可以贡献什么

- CANN、驱动、`torch_npu`、PyTorch 等版本组合的环境记录。
- 模型推理、LoRA 微调、profiling、Ascend C 算子的复现实验。
- FAQ、错误日志、排障路径和版本差异说明。
- Qwen 主线案例或视觉、多模态拓展案例。
- 文档修正、链接更新、目录整理。

## 开始前

1. 阅读 [CONTENT_GUIDE.md](CONTENT_GUIDE.md)。
2. 确认新增内容放在正确章节。
3. 如果是实操内容，先记录硬件、系统、CANN、Python、PyTorch 和 `torch_npu` 版本。
4. 如果涉及性能数据，写清模型、输入长度、batch、并发、测试命令和统计方式。

## 提交 PR 前

- 文档能在 GitHub 上正常阅读。
- 命令、输出、图片路径没有本地隐私信息。
- 代码和配置不包含 token、账号、内网地址。
- 主线实操不依赖多卡环境；多卡内容放在扩展说明或 FAQ。
- README、章节索引、FAQ 需要同步的地方已经同步。

