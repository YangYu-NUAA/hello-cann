# hello-cann 内容规范

这份规范用于 hello-cann 第一阶段共建，统一教程路径、版本、命令、输出和实验记录。

## 目录与命名

| 类型 | 推荐位置 | 示例 |
|:---|:---|:---|
| 环境教程 | `docs/zh/00_environment/` | `install-cann.md`、`torch-npu-check.md` |
| 推理教程 | `docs/zh/01_inference/` | `transformers-torch-npu.md`、`vllm-ascend.md` |
| 微调教程 | `docs/zh/02_finetune/` | `lora-single-card.md`、`dataset-format.md` |
| 性能分析 | `docs/zh/03_profiling/` | `profiling-quickstart.md`、`report-template.md` |
| Ascend C | `docs/zh/04_ascendc/` | `vector-add.md`、`operator-template.md` |
| 贯穿案例 | `docs/zh/05_cases/` | `qwen-optimization.md` |
| 参考资料 | `docs/zh/06_reference/` | `faq.md`、`version-matrix.md`、`components.md` |
| 案例工程 | `cases/<case-name>/` | `cases/qwen/` |
| 示例代码 | `src/<chapter>/` | `src/01_inference/`、`src/04_ascendc/` |
| 章节 Notebook | `notebooks/` | `01_inference.ipynb`、`02_finetune.ipynb` |
| 图片资源 | `assets/<chapter>/` | `assets/03_profiling/` |

文件名使用小写英文、数字和连字符。目录名使用课程编号加英文名，例如 `00_environment`。

## 单篇教程结构

每篇教程不必硬套标题，但这些信息要尽量完整：

1. 本节目标：这一节要跑通什么，最终输出是什么。
2. 测试环境：硬件、系统、驱动、固件、CANN、Python、PyTorch、`torch_npu` 和相关工具版本。
3. 前置条件：读者需要先完成哪些章节，需要哪些模型权重、数据或权限。
4. 操作步骤：命令、配置、关键输出，按实际执行顺序写。
5. 验证方式：怎么确认推理、训练、转换或算子运行成功。
6. 常见问题：记录本次实践中遇到过的坑。
7. 后续阅读：链接到官方文档、源码或本仓库其他章节。

推荐在文档开头放环境表：

```markdown
| 项目 | 版本或说明 |
|:---|:---|
| 硬件 | Atlas / Ascend 型号，卡数，单卡显存 |
| 系统 | Ubuntu / openEuler 版本 |
| 驱动与固件 | 版本号 |
| CANN | 版本号 |
| Python | 版本号 |
| PyTorch | 版本号 |
| torch_npu | 版本号 |
```

## 实操边界

第一阶段实操默认按单卡环境设计。

- LoRA 微调写单卡流程，训练日志和权重验证必须可复现。
- 多卡训练、tensor parallel、HCCL 放在概念说明、FAQ 和报错索引里。
- 如果贡献者有多卡环境，可以提交扩展记录，但不要让主线教程依赖多卡资源。
- 性能对比必须写清输入、batch、seq length、并发数、模型版本和测试命令。

## 命令与输出

- Shell 命令使用 `bash` 代码块。
- Python 代码使用 `python` 代码块。
- 输出不必完整贴满，保留能判断成功或定位失败的关键行。
- 不要把本地私有路径、账号、token、内网地址写进文档。

示例：

```bash
python - <<'PY'
import torch
import torch_npu

print(torch.__version__)
print(torch_npu.__version__)
print(torch.npu.is_available())
PY
```

如果某个命令只在特定版本可用，需要在命令前说明适用范围。

## 图片与截图

- 图片放到 `assets/` 下对应章节目录。
- Markdown 中用相对当前文件的路径引用图片。
- 截图优先展示关键输出、配置、profile 结果，不放装饰图。
- 图片文件名用小写英文、数字和连字符，例如 `vllm-server-started.png`。

## Notebook

- 每章在完成首轮服务器实验后提交一个 Notebook。
- Notebook 调用 `cases/` 或 `src/` 中的脚本，不复制另一套推理、训练或算子实现。
- 模型路径、数据路径和输出路径集中放在开头的参数单元。
- 提交前清空运行输出，真实指标另存到 `cases/<case-name>/reports/`。
- 下载大模型、合并完整权重等占用较大的操作默认关闭，并在单元前说明。
- Notebook 中不保存 token、账号、内网地址和私有数据。

## 写作自检

- 少写抽象评价，多写硬件、版本、命令、输出和指标。
- 不只写成功路径，安装、权限、版本、算子、HCCL 报错都值得记录。
- 只保留当前实验需要的概念，扩展内容放到对应参考页。
- 没有实际测试的数据，不写成确定结论。
- 性能提升必须有对比条件；没有对比就写“本节暂不比较性能”。

## 提交前自检

- [ ] 文档路径符合目录规划。
- [ ] 环境表写清楚了硬件、系统和版本。
- [ ] 命令在文中给出的环境下跑过。
- [ ] 至少有一种明确的成功验证方式。
- [ ] 图片路径真实存在，且能在 GitHub 中显示。
- [ ] 本章 Notebook 与脚本使用同一组参数和路径。
- [ ] 没有本地绝对路径、账号、token 或隐私信息。
- [ ] README、课程大纲、章节索引需要同步的地方已经检查。

规范会随着第一批教程继续调整。
