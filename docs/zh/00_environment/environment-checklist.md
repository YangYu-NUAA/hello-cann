# 环境检查清单

## 测试环境

| 项目 | 版本或说明 |
|:---|:---|
| 硬件 |  |
| 系统 |  |
| 驱动与固件 |  |
| CANN |  |
| Python |  |
| PyTorch |  |
| torch_npu |  |

## 检查项

- [ ] `npu-smi info` 能看到 NPU。
- [ ] CANN Toolkit 路径存在。
- [ ] `set_env.sh` 已加载。
- [ ] Python 环境可用。
- [ ] PyTorch 可导入。
- [ ] `torch_npu` 可导入。
- [ ] 最小 NPU 张量计算能运行。

## 常用命令

```bash
npu-smi info
```

```bash
python - <<'PY'
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu available:", torch.npu.is_available())
PY
```

