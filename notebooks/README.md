# 章节 Notebook

| Notebook | 内容 | 对应正文 |
|:---|:---|:---|
| [00-environment-check.ipynb](00-environment-check.ipynb) | 检查 `torch_npu` 并运行最小 NPU 张量计算 | [00 环境配置](../docs/zh/00-environment/README.md) |
| [01-qwen-inference.ipynb](01-qwen-inference.ipynb) | 使用 Transformers 运行 Qwen 单卡推理 | [01 模型推理](../docs/zh/01-inference/README.md) |
| [02-qwen-lora.ipynb](02-qwen-lora.ipynb) | 检查训练数据并运行单卡 LoRA | [02 大模型微调](../docs/zh/02-fine-tune/README.md) |

## 启动

在仓库根目录加载 CANN 和 Python 环境，设置模型路径，然后启动 Jupyter：

```bash
source /path/to/cann/set_env.sh
source .venv/bin/activate
export MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
python -m jupyter lab
```

`01-qwen-inference.ipynb` 和 `02-qwen-lora.ipynb` 都读取 `MODEL_PATH`。使用 Hugging Face 缓存目录时，将它设为包含 `config.json` 和模型权重的本地目录。

03 章之后的 Notebook 会随对应实验一起提交。
