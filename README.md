# hello-cann

hello-cann 是一门面向昇腾 CANN 的开源实践课程，内容包括环境配置、模型推理、单卡 LoRA 微调、性能分析、Ascend C 算子开发和应用接入。

课程示例主要使用 Qwen 系列模型。除特别标注外，实验均可在单张昇腾 NPU 上完成。

## 开始学习

按章节顺序阅读正文。已经完成实验的章节同时提供 Notebook。

| 章节 | 正文入口 | Notebook |
|:---|:---|:---|
| 00 环境配置 | [环境检查与 `torch_npu` 验证](docs/zh/00-environment/README.md) | [00-environment-check.ipynb](notebooks/00-environment-check.ipynb) |
| 01 模型推理 | [Transformers 单卡推理](docs/zh/01-inference/README.md) | [01-qwen-inference.ipynb](notebooks/01-qwen-inference.ipynb) |
| 02 单卡微调 | [Qwen LoRA 微调](docs/zh/02-fine-tune/README.md) | [02-qwen-lora.ipynb](notebooks/02-qwen-lora.ipynb) |
| 03 性能分析 | [Profiling 与瓶颈定位](docs/zh/03-profiling/README.md) | - |
| 04 算子开发 | [Ascend C 自定义算子](docs/zh/04-ascend-c/README.md) | - |
| 05 综合案例 | [模型优化与应用接入](docs/zh/05-cases/README.md) | - |
| 06 参考资料 | [版本、组件与常见问题](docs/zh/06-references/README.md) | - |

完整目录见 [中文课程目录](docs/zh/README.md)，课程规划见 [COURSE_OUTLINE.md](COURSE_OUTLINE.md)。

## 使用方式

克隆仓库后，从第 00 章开始阅读：

```bash
git clone https://github.com/YangYu-NUAA/hello-cann.git
cd hello-cann
```

正文中的模型目录通过 `MODEL_PATH` 指定。模型保存在什么位置都可以，例如：

```bash
export MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
```

使用已下载的权重时在命令中加入 `--local-files-only`；也可以把 `--model` 设为 Hugging Face 模型名称。

Notebook 和 Markdown 使用同一组脚本。Notebook 适合逐步执行，Markdown 适合在服务器终端中复制命令。

## 学习内容

1. 检查 NPU、驱动、CANN、Python、PyTorch 和 `torch_npu`。
2. 使用 Transformers 在昇腾 NPU 上运行 Qwen 推理。
3. 完成单卡 LoRA 微调，保存 adapter 并比较微调前后的输出。
4. 采集 profile，查看算子耗时、显存和数据搬运。
5. 编写 Ascend C 算子，完成编译、正确性检查和性能记录。
6. 将优化后的模型服务接入上层应用。

多卡训练、tensor parallel 和 HCCL 作为扩展内容介绍，不是完成课程的必需条件。

## 仓库结构

```text
hello-cann/
├── docs/zh/                  # 中文课程正文
│   ├── 00-environment/
│   ├── 01-inference/
│   ├── 02-fine-tune/
│   ├── 03-profiling/
│   ├── 04-ascend-c/
│   ├── 05-cases/
│   └── 06-references/
├── notebooks/                # 可直接运行的章节 Notebook
├── cases/                    # Qwen 案例脚本、配置和实验记录
├── src/                      # 环境检查脚本和 Ascend C 工程
├── assets/                   # 图片与性能分析结果
└── templates/                # 实验记录和案例模板
```

## 已完成内容

- 环境检查、CANN 环境加载和最小 NPU 张量计算。
- Qwen2.5-0.5B-Instruct Transformers 单卡推理。
- Qwen2.5-0.5B-Instruct 单卡 LoRA 训练、验证集评估和 adapter 对比。

实验使用的软件版本和设备信息集中记录在 [版本矩阵](docs/zh/06-references/version-matrix.md)。

## 贡献

欢迎提交教程、脚本、算子、实验记录和排障说明。新增内容请参考 [CONTENT_GUIDE.md](CONTENT_GUIDE.md)。

## License

本项目采用 MIT License。
