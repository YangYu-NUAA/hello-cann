# Notebooks

Notebook 用于运行每章的主要实验，章节正文和排障说明仍放在 `docs/zh/`。

| Notebook | 对应章节 | 状态 |
|:---|:---|:---|
| `00_environment.ipynb` | 环境与版本基线 | 已实测 |
| `01_inference.ipynb` | Transformers 单卡推理 | 已实测 |
| `02_finetune.ipynb` | 单卡 LoRA 微调 | 已实测 |

03 至 06 章在完成首轮服务器实验后补充 Notebook，不提前放不能运行的占位代码。

## 启动

先在终端加载 CANN 环境并激活虚拟环境，再启动 Jupyter：

```bash
source /path/to/cann/set_env.sh && source .venv/bin/activate && python -m jupyter lab
```

Notebook 中的模型路径默认为 `/mnt/workspace/data/Qwen2.5-0.5B-Instruct`。其他环境需要修改路径单元。
