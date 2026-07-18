# hello-cann

hello-cann 是一门面向昇腾 CANN 的开源实战课程。

课程从一台可用的昇腾机器开始，依次完成环境检查、模型推理、单卡 LoRA 微调、profiling、瓶颈定位、Ascend C 算子开发和模型接入。主线案例使用 Qwen 系列模型，尽量让每一章都有能复跑的命令、输出和记录模板。

课程实操以单卡环境为准。多卡训练、tensor parallel 和 HCCL 放在软件栈说明、FAQ 和报错索引里，不影响主线学习。

第一组实验环境固定在一台 IT22HMDA_4_S 双芯片昇腾机器上，总显存 128 GB，系统为 Ubuntu 20.04.5 LTS aarch64，驱动版本为 25.5.5，CANN 版本为 9.0.0，Python 版本为 3.11.4。

## 适合谁

- 有 Python、PyTorch 或大模型应用基础，希望上手昇腾 NPU 的开发者。
- 想系统学习 CANN、`torch_npu`、Ascend C、自定义算子和模型优化的学生、工程师和研究人员。
- 负责模型迁移、推理服务、训练优化、算子开发或性能调优的工程开发者。
- 对算子开发、算子竞赛、开源算子贡献感兴趣的社区开发者。

## 学习路线

1. **Environment**：检查硬件、驱动、固件、CANN、Python、PyTorch 和 `torch_npu`。
2. **Inference**：用 Transformers、vLLM-Ascend 或 MindIE 跑通 Qwen 系列模型推理。
3. **Fine-tune**：完成一次单卡 LoRA 微调，保存日志、权重和验证结果。
4. **Profiling**：观察推理或训练链路，定位热点算子、显存和数据搬运问题。
5. **Ascend C**：从基础算子开始，理解 tiling、数据搬运、编译运行和调试。
6. **Cases**：把推理、微调、profiling、算子开发、模型接入和应用接入串成一个项目。
7. **Reference**：整理版本矩阵、FAQ、术语表、组件索引和常见错误。

## 仓库结构

```text
hello-cann/
├── docs/
│   └── zh/
│       ├── 00_environment/   # 环境与版本基线
│       ├── 01_inference/     # 模型推理与服务化
│       ├── 02_finetune/      # 单卡 LoRA 微调
│       ├── 03_profiling/     # 性能分析与瓶颈定位
│       ├── 04_ascendc/       # Ascend C、自定义算子和 LLM 常用算子
│       ├── 05_cases/         # 贯穿案例和应用项目接入
│       └── 06_reference/     # 资料、FAQ 和术语表
├── cases/                    # 可复现案例工程
├── src/                      # 每章配套代码
├── notebooks/                # 交互式教程
├── assets/                   # 架构图、截图、profile 结果、性能表
├── templates/                # 教程、实验记录、案例报告模板
├── COURSE_OUTLINE.md         # 课程大纲
├── COMMUNITY_TASK.md         # 社区任务描述
└── CONTENT_GUIDE.md          # 内容规范
```

## 当前阶段

第一阶段先把 Qwen 主线跑稳。服务器实验按“先能复现，再补优化”的顺序推进：

- 环境检查：`npu-smi`、驱动、CANN、HCCL 和 `torch_npu`；Docker 按平台条件选做。
- 快速推理：Transformers + `torch_npu` 跑通 Qwen 小模型。
- 基线记录：固定 prompt、输出长度和版本，留下 JSON 结果。
- 单卡微调：Qwen LoRA 先做 smoke test，再补完整训练记录。
- 性能分析：用同一条推理命令采集 profile，整理热点表。
- 算子开发：先编译和验证 Vector Add，再选择一个 LLM 常用算子继续做。
- 服务化推理：vLLM-Ascend 或 MindIE 二选一跑通 OpenAI 兼容接口。
- 应用接入：模型服务稳定后，再接 `hello-agent`、`hello-claw` 或 `torch-rechub`。
- 案例报告：整理版本、命令、指标、问题和可复现步骤。

## 文档写法

每篇教程都要写清楚：

- 在什么硬件、系统、CANN 和框架版本下验证过。
- 执行了哪些命令，关键输出是什么。
- 怎么判断任务真的成功。
- 遇到问题时先查哪里。
- 本节是否做性能对比；如果做，指标和输入条件是什么。

新增内容前先看 [CONTENT_GUIDE.md](CONTENT_GUIDE.md)。

## 贡献

欢迎提交教程、脚本、排障记录和实验结果。跑通了可以写，失败记录也可以写，只要能帮助后面的学习者少踩一次坑。

可以先从这些内容开始贡献：

- 某个 CANN / `torch_npu` 版本组合的环境记录。
- Qwen 推理、LoRA、profiling 的复现实验。
- Ascend C 基础算子或调试记录。
- 一个 LLM 常用算子的 CANN 实现、正确性测试和性能记录。
- 昇腾模型服务接入上层应用的实验记录。
- FAQ、错误日志和排查路径。
- 案例报告和性能记录表。

## License

本项目采用 MIT License。
