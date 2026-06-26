# hello-cann 课程大纲

这份大纲用于 hello-cann 课程的初步设计。后续会随着机器环境、模型选择和案例验证继续调整。

## 课程定位

hello-cann 是一门面向昇腾 CANN 的实战课程。课程从一台可用的昇腾机器开始，依次覆盖环境配置、模型推理、LoRA 微调、profiling、瓶颈定位、Ascend C 算子开发和模型接入。

课程主线放在 PyTorch + `torch_npu`，配套介绍 ACL、ATC/OM、GE、HCCL、MindSpeed 等组件在具体任务中的位置。课程重点放在版本匹配、NPU 运行时、profiling、Ascend C、算子接入和性能验证。

课程实操以单卡环境为准。多卡训练、tensor parallel 和 HCCL 放在软件栈说明、FAQ 和报错索引里。

## 组件出现位置

第一阶段放一张软件栈关系图，说明各组件所在层次、使用场景和常见排障入口。

| 位置 | 内容 | 写到什么程度 |
|:---|:---|:---|
| 主线展开 | `torch_npu`、profiling、Ascend C、自定义算子接入 | 写步骤、代码、验证和排障 |
| 章节穿插 | ACL / AscendCL、ATC / OM、GE / GraphEngine、HCCL | 在用到时讲清用途和常见问题 |
| 进阶参考 | `cann-ops-adv`、MindSpeed、MindSpeed-LLM、MindSpore、DVPP / AIPP | 放参考资料、扩展阅读或第二阶段案例 |

GE 放在图编译、图优化和算子接入报错里；HCCL 放在通信概念、软件栈关系和常见报错索引里；`cann-ops-adv` 放在 Ascend C 之后，作为融合算子工程参考。

## 读者走完后应该具备的能力

- 能检查昇腾机器、驱动、固件、CANN、Python、PyTorch 和 `torch_npu` 是否匹配。
- 能用 Transformers、vLLM-Ascend 或 MindIE 跑通一个大模型推理服务。
- 能完成一次 LoRA 微调，并保存训练日志、权重和验证结果。
- 能用 profiling 工具找到模型推理或训练中的主要瓶颈。
- 能写出一个基础 Ascend C 算子，理解 tiling、数据搬运和编译运行流程。
- 能把自定义算子接入模型链路，并用指标说明优化是否有效。
- 能整理一份可复现的实验记录，给别人复跑或评审。

## 仓库结构

| 目录 | 内容 | 说明 |
|:---|:---|:---|
| `docs/zh/00_environment` | 环境与版本基线 | 硬件检查、CANN 安装、`torch_npu` 验证、Docker 使用 |
| `docs/zh/01_inference` | 模型推理与服务化 | Transformers、vLLM-Ascend、MindIE、benchmark |
| `docs/zh/02_finetune` | 微调与训练 | 数据准备、单卡 LoRA、训练日志、权重验证 |
| `docs/zh/03_profiling` | 性能分析 | profiling 工具、热点定位、显存、数据搬运和通信概念 |
| `docs/zh/04_ascendc` | Ascend C 与自定义算子 | 基础算子、tiling、融合、调试、模型接入 |
| `docs/zh/05_cases` | 贯穿案例 | Qwen 训推优化、视觉/多模态推理优化 |
| `docs/zh/06_reference` | 资料与排障 | 版本矩阵、FAQ、术语表、组件索引、链接索引 |
| `src/` | 示例代码 | 每章可运行代码、算子工程、脚本 |
| `notebooks/` | 交互式教程 | 适合 notebook 展示的章节和练习 |
| `assets/` | 图片与结果 | 架构图、截图、profile 结果、性能表 |

## 课程模块

| 模块 | 主题 | 完成后应该会什么 | 主要产出 |
|:---|:---|:---|:---|
| 00 Environment | 环境与版本基线 | 会确认机器是否能继续做 CANN 开发 | 环境检查清单、版本矩阵、排障记录 |
| 01 Inference | 大模型推理与服务化 | 会把模型跑起来，并记录基本性能指标 | Qwen 推理教程、服务启动脚本、benchmark 表 |
| 02 Fine-tune | 微调与训练 | 会准备数据、跑 LoRA、保存权重并验证 | 训练脚本、日志模板、结果说明 |
| 03 Profiling | 性能分析 | 会看 profile，能说清楚主要耗时在哪里 | profile 报告、热点算子记录 |
| 04 Ascend C | 自定义算子 | 会写基础算子，并理解接入模型的流程 | 算子代码、编译脚本、正确性测试 |
| 05 Cases | 综合案例 | 会把部署、训练、分析和优化合成一个项目 | 案例代码、优化报告、复现实验表 |
| 06 Reference | 资料与 FAQ | 能快速查版本、术语、常见错误 | FAQ、术语表、组件索引、链接索引 |

## 详细章节

### 00. Environment：先把机器变成可教学环境

这一章的目标很简单：读者照着做完，能判断自己的环境是否能继续往下走。

- 00.1 昇腾硬件与软件栈速览：NPU、驱动、固件、CANN、`torch_npu`、ACL、ATC/OM、GE、HCCL 之间的关系。
- 00.2 系统与硬件检查：`npu-smi info`、驱动版本、CANN 路径、设备权限。
- 00.3 CANN 安装与环境变量：Toolkit、Kernel、`set_env.sh`、多版本切换。
- 00.4 Python 环境：PyTorch、`torch_npu`、可选 MindSpore，安装后做最小验证。
- 00.5 Docker 使用：镜像选择、设备挂载、数据目录、权限问题。
- 00.6 常见问题：版本不匹配、找不到算子、环境变量丢失、`import torch_npu` 失败。

首篇教程以版本表、检查命令、关键输出和几条真实排障经验为主。

### 01. Inference：把第一个大模型服务跑起来

这一章负责让读者从命令行走到稳定服务。

- 01.1 Transformers + `torch_npu`：最小文本生成，确认模型、tokenizer、NPU 都在工作。
- 01.2 vLLM-Ascend 快速部署：启动 OpenAI 兼容服务，跑一次请求，记录显存占用。
- 01.3 MindIE 部署：生产推理常见流程、配置文件和启动方式。
- 01.4 推理扩展概念：batch、并发、显存占用、tensor parallel 与多卡场景的基本概念。
- 01.5 模型专题：优先用 Qwen 系列，后续再补 DeepSeek-R1-Distill、GLM、InternVL 或 Qwen-VL。
- 01.6 benchmark 模板：固定 prompt、并发数、输出长度、吞吐、延迟和显存记录。

有较充足的昇腾显存，可以安排两层模型：小模型用于快速验证，中等规模模型用于展示真实推理服务。教程里要说明两者用途，区分最小验证和完整服务体验。

### 02. Fine-tune：从能推理到能改模型

这一章跑通一次可复现的 SFT 或 LoRA 微调。

- 02.1 数据准备：Alpaca、ShareGPT 或自定义 JSONL 格式，训练/验证切分。
- 02.2 单卡 LoRA：以 Qwen 系列为主线，记录 batch、seq length、显存占用。
- 02.3 多卡训练概念：HCCL、进程启动方式和常见通信报错，放入扩展阅读和 FAQ 索引。
- 02.4 训练监控：loss 曲线、吞吐、显存、日志保存。
- 02.5 权重合并与推理验证：训练后如何确认模型真的学到了东西。
- 02.6 工具线选择：微调工具接入方式、训练脚本结构和结果验证。

这一章优先写“在哪台机器、哪组版本、哪条命令跑过”。失败记录可以放进 FAQ，后续很有用。

### 03. Profiling：让读者看懂模型为什么慢

这一章是从“会跑”到“会分析”的过渡。

- 03.1 profiling 工具概览：PyTorch profiler、CANN profiling、msProf 的适用场景。
- 03.2 推理链路拆解：prefill、decode、Host 下发、Device 执行、数据搬运。
- 03.3 热点算子定位：耗时、调用次数、shape、format、内存读写。
- 03.4 显存与通信概念：显存峰值、碎片、Host/Device 数据搬运、HCCL 常见报错线索。
- 03.5 图编译相关线索：ATC/OM、GE、算子支持和 shape 问题在 profile 与报错中的表现。
- 03.6 报告模板：用表格记录优化前后的吞吐、延迟、显存和正确性。

每个 profile 结果都要回答三个问题：瓶颈在哪里，为什么会这样，下一步准备怎么验证。

### 04. Ascend C：从第一个算子到模型接入

这一章让读者第一次真正摸到 CANN 的底层开发。

- 04.1 CANN 软件栈关系：应用、框架、图编译、运行时、算子、驱动之间的边界；GE、ACL、ATC/OM 讲到定位问题所需的边界。
- 04.2 Ascend C 入门：Vector Add，host 侧、kernel 侧、编译与运行。
- 04.3 Tiling 与数据搬运：Global Memory、Local Memory、队列、流水。
- 04.4 MatMul 与融合算子：从基础算子理解到可优化位置。
- 04.5 自定义算子工程化：目录结构、编译脚本、单元测试、正确性校验。
- 04.6 接入 PyTorch / `torch_npu`：从最小调用到模型链路中的替换。
- 04.7 调试与性能优化：profile、边界 case、精度误差、性能回退。
- 04.8 融合算子参考：选择性阅读 `cann-ops-adv` 中的实现，看工程结构和优化思路。

这部分是 hello-cann 和普通大模型教程拉开差异的地方。写法要慢一点，先把一件小事跑透，再讲为什么这么写。

### 05. Cases：把教程变成项目

这一章沉淀可以展示、可以复跑、可以继续改的案例。

主线案例使用 Qwen 系列模型：

- 环境检查：确认 NPU、CANN、`torch_npu` 和依赖版本。
- 推理部署：跑通 Qwen 小模型，再按机器条件扩展到更大模型。
- LoRA 微调：准备一份小数据集，完成训练和推理验证。
- profiling：记录优化前的吞吐、延迟、显存和热点算子。
- 自定义算子：选择一个教学友好的热点，完成 Ascend C 实现。
- 模型接入：把算子接回推理链路，做正确性和性能对比。
- 案例复盘：整理代码结构、版本、指标、问题和下一步优化方向。

拓展案例可以选视觉、视频或多模态模型。第一阶段先准备统一的性能记录模板，后续继续补完整实验。

### 06. Reference：资料、术语与排障索引

这一章负责把分散信息变成能查的东西。

- 官方文档入口：CANN、Ascend C、ACL、ATC、`torch_npu`、MindIE、MindSpore。
- 版本矩阵：CANN、驱动固件、Python、PyTorch、`torch_npu`、vLLM-Ascend。
- 常见错误：安装、运行时、模型加载、算子、HCCL、显存、权限。
- 组件索引：`torch_npu`、ACL、ATC/OM、GE、HCCL、`cann-ops-adv`、MindSpeed。
- 术语表：NPU、AI Core、AICPU、ACL、OM、GE、HCCL、tiling、workspace。
- 社区项目：vLLM-Ascend、Ascend/pytorch、Ascend samples、ModelZoo。

## 参考链接

- 昇腾文档中心：https://www.hiascend.com/document
- CANN 社区版资源中心：https://www.hiascend.com/developer/download/community/result?module=cann
- Ascend Extension for PyTorch：https://github.com/Ascend/pytorch
- vLLM-Ascend：https://github.com/vllm-project/vllm-ascend
- Ascend samples：https://github.com/Ascend/samples
- Ascend ModelZoo-PyTorch：https://github.com/Ascend/ModelZoo-PyTorch
