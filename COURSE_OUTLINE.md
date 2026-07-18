# hello-cann 课程大纲

这份大纲记录 hello-cann 第一版课程的主线、产出和验收方式。每节都会尽量写清楚要做什么、做到什么程度、最后留下哪些结果，方便后续逐章实验和补正文。

大纲会随着机器环境、模型选择和案例验证继续调整，调整时请同步更新本文件，而不是只改章节正文。

---

## 课程定位

hello-cann 是一门面向昇腾 CANN 的实战课程。课程从一台可用的昇腾机器开始，依次覆盖环境配置、模型推理、LoRA 微调、profiling、瓶颈定位、Ascend C 算子开发和模型接入。

课程主线放在 PyTorch + `torch_npu`，配套介绍 ACL、ATC/OM、GE、HCCL、MindSpeed 等组件在具体任务中的位置。课程重点放在版本匹配、NPU 运行时、profiling、Ascend C、算子接入和性能验证。

课程实操以单卡环境为准。多卡训练、tensor parallel 和 HCCL 放在软件栈说明、FAQ 和报错索引里，不影响主线学习。

### 课程特点

- 每章都围绕可复现实验展开，命令、版本、输出和问题记录要能留在仓库里。
- 推理、微调、profiling、算子开发尽量使用同一条 Qwen 主线，减少读者在不同案例之间来回切换。
- 多卡训练、tensor parallel 和 HCCL 先放在概念说明和排障索引里，不作为第一版必做实验。
- Ascend C 先跑通 Vector Add，再选择一个 LLM 常用算子做正确性和性能对比；其余算子作为扩展材料逐步补。

### 第一组实验环境

| 项目 | 值 |
|:---|:---|
| 昇腾卡型号 | IT22HMDA_4_S（2 芯片） |
| 单芯片显存 | 64 GB HBM（共 128 GB） |
| 操作系统 | Ubuntu 20.04.5 LTS，kernel 5.10.0-182（aarch64） |
| 驱动版本 | 25.5.5（Innerversion: V100R001C23SPC009B220） |
| 固件版本 | `npu-smi` 返回 `NA` |
| CANN 版本 | 9.0.0（`~/Ascend/cann-9.0.0/`） |
| Python 版本 | 3.11.4 |
| PyTorch / torch_npu | torch 2.7.1+cpu / torch_npu 2.7.1.post2.dev20251226 |

00 章已经完成实测：HCCL 动态库可用，`torch_npu` 能识别 2 个 NPU，最小张量计算通过。Transformers 在 01 章安装。

### 第一轮实验边界

第一轮服务器实验先做最小闭环：

1. 环境检查：`npu-smi`、CANN 路径、`torch_npu` 导入、最小 NPU 张量。
2. 推理 baseline：Transformers + `torch_npu` 跑通 Qwen2.5-0.5B，保存 JSON。
3. 本地 benchmark：只记录 `concurrency=1` 的基础吞吐、延迟和显存。
4. profiling：用同一条推理命令采集一次 profile，整理热点表。
5. LoRA：用示例数据跑一个 smoke test，确认训练、保存和合并流程。
6. Ascend C：先尝试编译 Vector Add，记录 CANN 路径、SoC 名称和报错。

vLLM-Ascend、MindIE、应用接入和 LLM 常用算子放在第二轮。第一轮不要求一次性完成所有章节。

### 组件出现位置

| 位置 | 内容 | 写到什么程度 |
|:---|:---|:---|
| 主线展开 | `torch_npu`、profiling、Ascend C、自定义算子接入 | 写步骤、代码、验证和排障 |
| 章节穿插 | ACL / AscendCL、ATC / OM、GE / GraphEngine、HCCL | 在用到时讲清用途和常见问题 |
| 应用接入 | vLLM-Ascend、MindIE、OpenAI 兼容接口、应用侧网关 | 放在推理服务和应用案例中 |
| 进阶参考 | `cann-ops-adv`、MindSpeed、MindSpeed-LLM、MindSpore、DVPP / AIPP | 放参考资料、扩展阅读或第二阶段案例 |

GE 放在图编译、图优化和算子接入报错里；HCCL 放在通信概念、软件栈关系和常见报错索引里；`cann-ops-adv` 放在 Ascend C 之后，作为融合算子工程参考。

### 读者走完后应该具备的能力

- 能检查昇腾机器、驱动、固件、CANN、Python、PyTorch 和 `torch_npu` 是否匹配。
- 能用 Transformers、vLLM-Ascend 或 MindIE 跑通一个大模型推理服务。
- 能完成一次 LoRA 微调，并保存训练日志、权重和验证结果。
- 能用 profiling 工具找到模型推理或训练中的主要瓶颈。
- 能写出一个基础 Ascend C 算子，理解 tiling、数据搬运和编译运行流程。
- 能从 PyTorch 参考实现出发，完成至少一个 LLM 常用算子的 CANN 实现和对比。
- 能把自定义算子接入模型链路，并用指标说明优化是否有效。
- 能把部署在昇腾上的模型服务接入一个上层应用，记录端到端任务耗时。
- 能整理一份可复现的实验记录，给别人复跑或评审。

---

## 仓库结构

| 目录 | 内容 | 说明 |
|:---|:---|:---|
| `docs/zh/00_environment` | 环境与版本基线 | 硬件检查、CANN 安装、`torch_npu` 验证、Docker 使用 |
| `docs/zh/01_inference` | 模型推理与服务化 | Transformers、vLLM-Ascend、MindIE、benchmark；Qwen 主线 |
| `docs/zh/02_finetune` | 微调与训练 | 数据、PEFT 原理、训练配置、权重合并、单卡 LoRA 实战 |
| `docs/zh/03_profiling` | 性能分析 | PyTorch profiler、CANN profiling、msProf、热点定位、报告 |
| `docs/zh/04_ascendc` | Ascend C 与自定义算子 | 五步入门、tiling、工程化、LLM 算子、模型接入 |
| `docs/zh/05_cases` | 贯穿案例 | Qwen 训推优化、应用项目接入、案例报告 |
| `docs/zh/06_reference` | 资料与排障 | 版本矩阵、FAQ、术语表、组件索引、链接索引 |
| `cases/` | 可复现案例工程 | Qwen 主线工程、扩展模型工程 |
| `src/` | 示例代码 | 每章可运行代码、算子工程、脚本 |
| `notebooks/` | 交互式教程 | 适合 notebook 展示的章节和练习 |
| `assets/` | 图片与结果 | 架构图、截图、profile 结果、性能表 |
| `templates/` | 文档模板 | 教程、实验记录、案例报告模板 |

---

## 课程模块总览

| 模块 | 主题 | 完成后应该会什么 | 主要产出 |
|:---|:---|:---|:---|
| 00 Environment | 环境与版本基线 | 会确认机器是否能继续做 CANN 开发 | 环境检查清单、版本矩阵、软件栈关系图、排障记录 |
| 01 Inference | 大模型推理与服务化 | 会把模型跑起来，并记录基本性能指标 | Qwen 推理教程、服务启动脚本、benchmark 表 |
| 02 Fine-tune | 微调与训练 | 会准备数据、跑 LoRA、保存权重并验证 | 训练脚本、日志模板、权重合并脚本、结果说明 |
| 03 Profiling | 性能分析 | 会看 profile，能说清楚主要耗时在哪里 | profile 报告、热点算子记录 |
| 04 Ascend C | 自定义算子 | 会写基础算子和 LLM 算子，并理解接入模型的流程 | 算子代码、编译脚本、正确性测试、性能记录 |
| 05 Cases | 综合案例 | 会把部署、训练、分析、优化和应用接入合成一个项目 | 案例代码、优化报告、复现实验表 |
| 06 Reference | 资料与 FAQ | 能快速查版本、术语、常见错误 | FAQ、术语表、组件索引、链接索引 |

---

## 详细章节

### 00. Environment：先把机器变成可教学环境

> **验收目标**：跑完本章后，读者能判断自己的环境是否可以继续做后面的推理、微调和算子实验，并跑通最小 NPU 张量校验。

**00.1 昇腾硬件与软件栈速览**
- 学习目标：建立 NPU、驱动、固件、CANN、`torch_npu`、ACL、ATC/OM、GE、HCCL 之间的层次认知。
- 主要产出：`docs/zh/00_environment/software-stack.md` + 一张软件栈关系图（`assets/00_environment/software-stack.{png,svg}`）。
- 前置依赖：无。
- 验证方式：读者能口述「一段 Python 代码 → torch_npu → CANN runtime → driver → NPU」的调用链。
- 写作要点：用一段 Python 推理代码串起「PyTorch → torch_npu → CANN runtime → driver → NPU」的调用链。

**00.2 系统与硬件检查**
- 学习目标：用 `npu-smi info`、驱动版本、CANN 路径、设备权限判断机器状态。
- 主要产出：`environment-checklist.md` 和 `src/00_environment/check_environment.sh`。
- 前置依赖：00.1。
- 验证方式：勾选清单全部通过。

**00.3 CANN 安装与环境变量**
- 学习目标：理解 Toolkit、Kernel、`set_env.sh` 的关系，能处理多版本切换。
- 主要产出：`install-cann.md`（新增）。
- 前置依赖：00.2。
- 验证方式：按实际路径加载 CANN 环境变量，或确认镜像已经预置；能读到 `ASCEND_HOME_PATH` 和安装版本信息。
- 写作要点：覆盖「`/usr/local/Ascend` 下只有 driver」的常见容器场景，给出 `find /usr/local/Ascend ~/Ascend -name set_env.sh` 的排查路径。

**00.4 Python 环境与 torch_npu 验证**
- 学习目标：安装并验证 PyTorch + `torch_npu`，跑通最小 NPU 张量计算。
- 主要产出：`torch-npu-check.md` 和环境检查脚本。
- 前置依赖：00.3。
- 验证方式：`torch.ones((2,3), device="npu")` 不报错。
- 排障重点：`libhccl.so` 查找、动态链接检查、`torch_npu` 导入错误和版本不匹配。

**00.5 Docker 使用（选做）**
- 学习目标：确认平台是否提供 Docker；具备条件时再使用昇腾镜像复现实验。
- 主要产出：`docker.md`。
- 前置依赖：00.3。
- 验证方式：容器内能跑通 00.4 的校验脚本。
- 写作要点：镜像选择、`--device /dev/davinci0` 挂载、数据目录、权限问题。

**00.6 常见问题**
- 学习目标：识别并处理版本不匹配、找不到算子、环境变量丢失、`import torch_npu` 失败。
- 主要产出：`troubleshooting.md`（新增，与 06 FAQ 区分——本章只放环境类）。
- 前置依赖：00.2-00.4。
- 验证方式：每条排障路径都给出「症状 → 原因 → 修复命令」三段式。

> **00 章验收清单（嵌 index.md）**
> - [ ] 能用 `npu-smi info` 看到 NPU 并读懂输出。
> - [ ] 已按实际路径加载 CANN 环境变量，或确认镜像已预置；`python -c "import torch_npu"` 不报错。
> - [ ] `libhccl.so` 可以被动态链接器找到；如果失败，已记录实际路径和环境变量。
> - [ ] 跑通最小 NPU 张量校验脚本，输出全 2 张量。
> - [ ] 已记录本机硬件、驱动、CANN、Python、PyTorch、torch_npu 版本到环境表。
> - [ ] 有一张能解释软件栈层次的关系图。

---

### 01. Inference：把第一个大模型服务跑起来

> **验收目标**：从命令行走到稳定 OpenAI 兼容服务，并留下可被 profiling 章复用的 baseline。

**目录布局（主线集中 + 扩展按模型）**：
- `docs/zh/01_inference/` 放 Qwen 主线（transformers / vllm / mindie / benchmark）。
- 扩展模型（DeepSeek-R1-Distill、GLM、InternVL、Qwen-VL）放 `cases/<model>/`，索引挂在本章 index。

**01.1 Transformers + torch_npu 最小推理**（已有）
- 学习目标：在昇腾 NPU 上跑一次 Qwen 文本生成，得到输出、JSON 记录、可复跑命令。
- 主要产出：`transformers-torch-npu.md` + `cases/qwen/scripts/run_transformers_torch_npu.py`（已有，保留）。
- 前置依赖：00 章验收通过。
- 验证方式：脚本输出 `tokens_per_second` + JSON 落盘到 `cases/qwen/results/`。

**01.2 vLLM-Ascend 服务化部署**
- 学习目标：启动 OpenAI 兼容服务，记录显存占用、并发能力。
- 主要产出：`vllm-ascend.md`（待实测后补启动命令）。
- 前置依赖：01.1。
- 验证方式：`curl` 发 chat completions 请求拿到流式响应。
- 写作要点：启动命令、`--port`、`--tensor-parallel-size`（单卡填 1）、显存占用对比表。
- 第一轮边界：不作为阻塞项；先完成 Transformers baseline。

**01.3 MindIE 部署**
- 学习目标：了解生产推理的另一种选择，掌握配置文件和启动方式。
- 主要产出：`mindie.md`（需确认获取方式和版本组合）。
- 前置依赖：01.1。
- 验证方式：MindIE 服务起来，能响应请求。
- 写作要点：与 vLLM-Ascend 的差异定位、适用场景、配置项说明。
- 第一轮边界：与 vLLM-Ascend 二选一，等公开获取方式和版本组合确认后再进入主线。

**01.4 推理扩展概念**
- 学习目标：理解 batch、并发、显存占用、tensor parallel 与多卡场景的基本概念。
- 主要产出：`concepts.md`（新增，或并入 benchmark.md）。
- 前置依赖：01.1。
- 边界：**只讲概念，不做多卡实战**（满足"不多卡实战"约束）；多卡实战放 FAQ 索引。

**01.5 模型专题与扩展**
- 学习目标：知道如何把主线方法迁移到其他模型。
- 主要产出：本章 index 的「扩展模型」表 + 各 `cases/<model>/README.md`。
- 优先级：Qwen 系列主线 → DeepSeek-R1-Distill → GLM → InternVL/Qwen-VL（视机器条件）。

**01.6 benchmark 模板**
- 学习目标：用固定 prompt、并发数、输出长度记录吞吐、延迟和显存，形成可对比的 baseline。
- 主要产出：`benchmark.md` + `cases/qwen/scripts/benchmark.py`（新增脚本）。
- 前置依赖：01.1。
- 验证方式：产出标准 benchmark 表（模型 / 输入长度 / batch / 并发 / 吞吐 / 延迟 / 显存）。
- 写作要点：区分本地脚本 benchmark 和服务化 benchmark；没有可比实验时，直接记录本次数据，不强行下结论。

> **01 章验收清单（嵌 index.md）**
> - [ ] Transformers + torch_npu 跑通 Qwen 小模型，JSON 结果落盘。
> - [ ] vLLM-Ascend 或 MindIE 至少一个服务化方案能响应 OpenAI 兼容请求。
> - [ ] 有一份 benchmark 表，字段齐全（模型/输入/batch/并发/吞吐/延迟/显存）。
> - [ ] baseline 命令可被 03 profiling 章直接复用。
> - [ ] 显存占用有真实数据，不是估计值。

---

### 02. Fine-tune：从能推理到能改模型

> **验收目标**：跑通一次可复现的单卡 LoRA 微调，保存训练日志、权重和推理验证结果。

**目录布局**：本章拆成「原理小节」+「实战小节」，主线用 Qwen，扩展模型放 `cases/<model>/`。

**原理小节**

**02.1 数据准备与格式**
- 学习目标：理解 Alpaca / ShareGPT / 自定义 JSONL，会做训练/验证切分。
- 主要产出：`dataset-format.md`（已有骨架，后续补样例）。
- 关键内容：Alpaca `instruction/input/output` 三字段；chat template 适配；`-100` label mask；MAX_LENGTH 截断。
- 写作要点：把 prompt 拼接、label mask 和截断逻辑写清楚，代码要能直接复用。

**02.2 PEFT 与 LoRA 原理**
- 学习目标：理解全量微调 vs 高效微调、LoRA 低秩分解、`r`/`alpha`/`dropout` 参数含义。
- 主要产出：`peft-principle.md`（新增）。
- 关键内容：`LoraConfig` 的 `target_modules`（Qwen 的 q/k/v/o/gate/up/down_proj）、`task_type=CAUSAL_LM`。

**02.3 训练配置与 Trainer**
- 学习目标：理解 `TrainingArguments` 的关键参数，会用 `Trainer` + `DataCollatorForSeq2Seq`。
- 主要产出：`training-config.md`（新增）。
- 关键内容：`per_device_train_batch_size`、`gradient_accumulation_steps`、`gradient_checkpointing`、`logging_steps`、`save_steps`、`report_to`。
- 边界：**单卡参数为主**，`save_on_each_node` 这种多机参数出现但不展开。

**实战小节**

**02.4 单卡 LoRA 实战（Qwen 主线）**
- 学习目标：在昇腾 NPU 上完成一次 Qwen LoRA 微调，记录 batch、seq length、显存、loss。
- 主要产出：`lora-single-card.md`（已有第一轮步骤）+ `cases/qwen/scripts/run_lora_sft.py`（新增脚本）+ `cases/qwen/configs/lora-sft.example.json`。
- 前置依赖：02.1-02.3。
- 验证方式：训练正常出 loss 曲线，checkpoint 落盘，显存占用有记录。

**02.5 训练监控与日志**
- 学习目标：用 SwanLab 记录 loss、吞吐、显存，保存训练日志。
- 主要产出：`training-log.md`（已有骨架）+ SwanLab 接入说明。
- 写作要点：训练监控先保证能记录 loss、耗时和显存；可视化平台作为可选项。

**02.6 权重合并与推理验证**
- 学习目标：合并 LoRA 权重回基座，用合并后的模型做推理验证。
- 主要产出：`weight-merge.md`（新增）+ `cases/qwen/scripts/merge_lora.py`（新增脚本）。
- 验证方式：合并前后输出对比，证明模型"学到了"任务（如角色扮演风格变化）。

**02.7 多卡训练概念（只讲不做）**
- 学习目标：知道 HCCL、进程启动方式、常见通信报错长什么样。
- 主要产出：本章末尾小节 + 06 FAQ 索引。
- 边界：**纯概念，无实战**，满足"不多卡实战"约束。

> **02 章验收清单（嵌 index.md）**
> - [ ] 02.1-02.3 三篇原理小节能独立读懂，不依赖实战代码。
> - [ ] 单卡 LoRA 训练跑通，loss 曲线、显存、checkpoint 都有记录。
> - [ ] 权重合并脚本可用，合并后模型推理输出符合预期。
> - [ ] 训练日志已落盘到 `cases/qwen/results/` 或 SwanLab。
> - [ ] 全程单卡，未出现必须多卡才能复现的步骤。

---

### 03. Profiling：让读者看懂模型为什么慢

> **验收目标**：对一次真实推理或训练采集 profile，定位主要耗时位置，并能用三个问题回答（瓶颈在哪 / 为什么 / 怎么验证）。

**03.1 profiling 工具概览**
- 学习目标：知道 PyTorch profiler、CANN profiling、msProf 各自的适用场景。
- 主要产出：`profiling-quickstart.md`（已有第一轮步骤）。
- 关键内容：三种工具的输入输出、适用阶段、产出格式对比表。

**03.2 推理链路拆解**
- 学习目标：拆解 prefill / decode / Host 下发 / Device 执行 / 数据搬运。
- 主要产出：`inference-breakdown.md`（新增）+ 链路时序图（`assets/03_profiling/timeline.{png,svg}`）。
- 写作要点：不要只贴截图，要解释时间主要花在哪里，以及这个结论还能不能复现。

**03.3 热点算子定位**
- 学习目标：从 profile 结果里找耗时 top-N 算子，看调用次数、shape、format、内存读写。
- 主要产出：`hotspot-operators.md`（新增）。
- 验证方式：产出热点算子表，每条标注是否可融合 / 是否可替换为自定义算子。
- **承接作用**：本章结果直接作为 04 章选择算子的依据。

**03.4 显存与通信概念**
- 学习目标：看懂显存峰值、碎片、Host/Device 数据搬运、HCCL 报错线索。
- 主要产出：`memory-and-communication.md`（新增）。
- 边界：通信只讲概念和报错识别，不做多卡实战。

**03.5 图编译相关线索**
- 学习目标：理解 ATC/OM、GE、算子支持和 shape 问题在 profile 与报错中的表现。
- 主要产出：`graph-compilation.md`（新增）。
- 关键内容：图编译耗时占比、算子不支持时的 fallback 行为。

**03.6 性能报告模板**
- 学习目标：用表格记录优化前后的吞吐、延迟、显存和正确性。
- 主要产出：`report-template.md`（已有，需对齐字段）。
- 验证方式：模板字段与 benchmark.md、case-report-template.md 一致。

> **03 章验收清单（嵌 index.md）**
> - [ ] 至少用一种工具（PyTorch profiler / CANN profiling / msProf）跑出 profile 结果。
> - [ ] 有一张推理链路时序图或截图。
> - [ ] 产出热点算子表，标注 top-3 耗时算子。
> - [ ] 每个结论都能回答「瓶颈在哪 / 为什么 / 下一步怎么验证」。
> - [ ] profile 用的命令与 01 baseline 一致，可复现。

---

### 04. Ascend C：从第一个算子到模型接入

> **验收目标**：会写基础 Ascend C 算子，理解 tiling、数据搬运、编译运行流程，并至少完成一个 LLM 常用算子的 CANN 实现与对比。

**入门路径（先小程序，再完整算子）**

**04.1 CANN 软件栈与算子开发边界**
- 学习目标：理清应用、框架、图编译、运行时、算子、驱动之间的边界；GE/ACL/ATC 讲到定位问题所需。
- 主要产出：`software-stack.md`（新增，与 00.1 互补——00 讲全景，04 聚焦算子层）。

**04.2 Hello World：Host 调 Device**
- 学习目标：跑通第一个 Ascend C 程序，理解 host 侧任务下发、device 侧执行。
- 主要产出：`hello-world.md`（新增）+ `src/04_ascendc/hello_world/`。
- 写作要点：把 host 侧下发和 device 侧执行分开讲，避免把同步、异步和算子执行混在一起。

**04.3 编程范式：CopyIn / Compute / CopyOut**
- 学习目标：理解 Global Memory、Local Memory、队列、流水。
- 主要产出：`programming-paradigm.md`（新增）。
- 写作要点：围绕数据从 GM 到 LM、再回到 GM 的过程讲清楚队列和流水。

**04.4 Tiling 入门**
- 学习目标：理解为什么要 tiling、tiling 参数怎么选、与数据搬运的关系。
- 主要产出：`tiling.md`（新增）+ tiling 决策图（`assets/04_ascendc/tiling-decision.{png,svg}`）。

**04.5 Vector Add 完整实战**
- 学习目标：跑通 Vector Add 的 host 侧、kernel 侧、编译、执行、正确性检查、profiling。
- 主要产出：`vector-add.md`（已有第一轮编译步骤）+ `src/04_ascendc/vector_add/` 工程三件套。
- 工程模板：`vector_add_kernel.cpp` + `vector_add_op.cpp`(aclnn) + `CMakeLists.txt`，由 msopgen 生成。
- 验证方式：与 PyTorch 参考实现 `torch.allclose` 对比通过。

**工程化与进阶**

**04.6 自定义算子工程化**
- 学习目标：掌握目录结构、编译脚本、单元测试、正确性校验的标准化做法。
- 主要产出：`operator-template.md`（已有骨架）。
- 写作要点：工程模板先服务于复现，不提前堆太多抽象。

**04.7 接入 PyTorch / torch_npu**
- 学习目标：从最小调用到模型链路中的算子替换。
- 主要产出：`model-integration.md`（待算子实测后补）。
- 关键技术：aclnn 调用、Python/C++ 封装、正确性测试；需要反向传播时再补 autograd。
- 写作边界：具体接入方式以当前 CANN 9.0.0 环境实测结果为准，不提前承诺未验证 API。

**04.8 LLM 常用算子的 CANN 实现**
- 学习目标：从 PyTorch 参考实现出发，先完成 RMSNorm、Softmax 或 SiLU/SwiGLU 中的一个，再按实验进展扩展。
- 主要产出：`llm-operators.md`（已有选题和记录模板）+ `src/04_ascendc/llm_ops/{rmsnorm,softmax,silu}/`。
- 素材来源：`llm-algo-leetcode` 中的 PyTorch/Triton/CUDA 题目结构（不写 CUDA，只借题目和测试方式）。
- 写作顺序建议：先 RMSNorm（最简单，规约+广播）→ Softmax（数值稳定性）→ SiLU/SwiGLU（接 Qwen FFN）。
- 每个算子的标准记录表：算子名 / shape / dtype / PyTorch 耗时 / CANN 耗时 / 最大误差 / 平均误差 / profile 结论 / 是否接入模型。

**04.9 调试与性能优化**
- 学习目标：profile、边界 case、精度误差、性能回退的排查方法。
- 主要产出：`debug-and-optimize.md`（新增）。

**04.10 融合算子参考（选读）**
- 学习目标：选择性阅读 `cann-ops-adv` 中的实现，看工程结构和优化思路。
- 主要产出：`fused-operators.md`（新增，选读指引）。

> **04 章验收清单（嵌 index.md）**
> - [ ] 跑通 Hello World 和 Vector Add 两个最小算子。
> - [ ] 有一套标准算子工程模板（msopgen 生成，CMake 编译）。
> - [ ] 至少完成一个 LLM 算子的 CANN 实现并通过正确性测试。
> - [ ] 每个算子有「PyTorch 参考 vs CANN 实现」的耗时和误差对比表。
> - [ ] 至少一个算子尝试接入模型链路（即便性能没赢，也要记录原因）。

---

### 05. Cases：把教程变成项目

> **验收目标**：把环境、推理、微调、profiling、算子开发和模型接入整理成可展示、可复跑、可继续改的完整项目。

**05.1 Qwen 训推与算子优化（主线案例）**
- 案例 7 步串法（已有框架，需补实操）：
  1. 环境检查（复用 00 产出）。
  2. 推理部署（复用 01 baseline）。
  3. LoRA 微调（复用 02 产出）。
  4. profiling（复用 03 产出，定位热点）。
  5. 自定义算子（复用 04 产出，选一个热点）。
  6. 模型接入（算子接回 Qwen 链路，正确性 + 性能对比）。
  7. 案例复盘（性能记录、代码结构、问题清单、复现实验表）。
- 主要产出：`qwen-optimization.md`（已有框架，需填实操）+ `cases/qwen/` 完整工程。

**05.2 昇腾模型服务接入应用项目**
- 学习目标：把部署在昇腾上的模型服务接到上层应用，记录端到端任务耗时。
- 主要产出：`application-projects.md`（已有框架）+ 各应用案例。
- 应用项目表：

| 项目 | 验证目标 | 产出形式 |
|:---|:---|:---|
| `hello-agent` | agent 任务能否稳定调用昇腾模型服务 | `cases/apps/hello-agent/` 实验记录 |
| `hello-claw` | OpenClaw / Skill 类应用切换到昇腾模型服务 | `cases/apps/hello-claw/` 实验记录 |
| `torch-rechub` / `rechub` | 推荐模型在 CANN 上的训练/推理/profiling | `cases/apps/torch-rechub/` 实验记录 |

- 应用指标表：首 token 延迟 / 单次响应耗时 / 应用任务耗时 / 模型调用次数 / 成功率 / 错误类型。

**05.3 视觉/多模态拓展案例（可选，第二阶段）**
- 案例 7 步同主线，模型换成视觉/视频/多模态。
- 第一阶段先预留接口和记录模板。

**05.4 案例报告模板**
- 主要产出：`case-report-template.md`（已有模板，待实测后对齐字段）。
- 字段与 benchmark.md、report-template.md 一致。

> **05 章验收清单（嵌 index.md）**
> - [ ] Qwen 主线案例至少包含：1 次推理部署 + 1 次微调 + 1 次 profiling + 1 次算子开发或接入 + 1 次优化前后对比。
> - [ ] 至少一个应用项目（hello-agent / hello-claw / torch-rechub）跑通并记录端到端耗时。
> - [ ] 案例报告字段齐全，可被他人复跑。
> - [ ] 复现实验表覆盖环境、版本、命令、指标、问题。

---

### 06. Reference：资料、术语与排障索引

> **验收目标**：把分散信息整理成能查的东西。

**06.1 版本矩阵**
- 主要产出：`version-matrix.md`（待持续更新）。
- 字段：CANN / 驱动固件 / Python / PyTorch / torch_npu / vLLM-Ascend / Transformers / PEFT。

**06.2 常见问题（FAQ）**
- 主要产出：`faq.md`（从真实问题里持续补）。
- 分类：安装 / 运行时 / 模型加载 / 算子 / HCCL / 显存 / 权限。
- 写法：「症状 → 原因 → 修复」三段式，每条标注首次出现的版本组合。

**06.3 组件索引**
- 主要产出：`components.md`（已有基础索引，后续补常见报错）。
- 组件：torch_npu / ACL / ATC-OM / GE / HCCL / cann-ops-adv / MindSpeed / MindSpeed-LLM / MindSpore。

**06.4 术语表**
- 主要产出：`glossary.md`（随正文术语持续补）。
- 术语：NPU / AI Core / AICPU / ACL / OM / GE / HCCL / tiling / workspace / DVPP / AIPP。

**06.5 链接索引**
- 主要产出：`links.md`（随引用持续补）。
- 分类：官方文档 / 社区项目 / 算子竞赛。

> **06 章验收清单（嵌 index.md）**
> - [ ] 版本矩阵覆盖课程用到的所有组件。
> - [ ] FAQ 收录第一轮实验里真实遇到的问题，每条三段式。
> - [ ] 组件索引每个组件都有「是什么 / 在哪用 / 常见报错」三行说明。
> - [ ] 术语表覆盖正文中出现的所有昇腾专有名词。
> - [ ] 链接全部有效；失效、需登录或需额外权限的链接单独标注。

---

## 参考链接

### 官方文档
- 昇腾文档中心：https://www.hiascend.com/document
- CANN 社区版资源中心：https://www.hiascend.com/developer/download/community/result?module=cann
- Ascend Extension for PyTorch：https://github.com/Ascend/pytorch
- vLLM-Ascend：https://github.com/vllm-project/vllm-ascend
- Ascend samples：https://github.com/Ascend/samples
- Ascend ModelZoo-PyTorch：https://github.com/Ascend/ModelZoo-PyTorch

### 算子竞赛与生态
- CANNJudge：https://cannjudge.cn
- cann-ops-adv：见 Ascend 官方仓库

### Datawhale 关联项目
- hello-agent：https://github.com/datawhalechina/hello-agent
- hello-claw：https://github.com/datawhalechina/hello-claw
- llm-algo-leetcode：https://github.com/datawhalechina/llm-algo-leetcode
- torch-rechub：https://github.com/datawhalechina/torch-rechub
- happy-llm：https://github.com/datawhalechina/happy-llm
