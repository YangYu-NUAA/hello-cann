# 环境检查清单

## 测试环境

| 项目 | 版本或说明 |
|:---|:---|
| 硬件 | IT22HMDA_4_S（2 芯片），64 GB HBM/芯片，共 128 GB |
| 系统 | Ubuntu 20.04.5 LTS，kernel 5.10.0-182（aarch64） |
| 驱动 | 25.5.1（Innerversion: V100R001C23SPC006B220） |
| 固件 | 不支持查询 |
| CANN | 9.1.0（`~/Ascend/cann-9.1.0/`，PATH 中已包含） |
| Python | 3.11.4 |
| PyTorch | torch 2.7.1+cpu |
| torch_npu | 2.7.1.post2.dev20251226（当前缺 `libhccl.so`） |
| Transformers | 未安装 |

## 检查项

- [ ] `npu-smi info` 能看到 NPU。
- [ ] CANN Toolkit 路径存在。
- [ ] CANN 环境变量已加载，或当前镜像已预置相关路径。
- [ ] `libhccl.so` 可以被动态链接器找到。
- [ ] Python 环境可用。
- [ ] PyTorch 可导入。
- [ ] `torch_npu` 可导入。
- [ ] Transformers 已安装。
- [ ] 最小 NPU 张量计算能运行。

## 常用命令

```bash
npu-smi info
```

查找 CANN 环境脚本：

```bash
find /usr/local/Ascend ~/Ascend -name set_env.sh 2>/dev/null
```

如果 `/usr/local/Ascend` 下面只有 `driver`，说明当前路径里只能看到驱动，不代表 toolkit 一定装在这里。继续检查 `~/Ascend`、镜像文档或管理员提供的 CANN 安装路径。

如果找到了环境脚本，再按实际路径加载：

```bash
source /path/to/set_env.sh
```

如果镜像已经预置了 CANN 环境变量，也可以直接进入后面的 Python 校验，并把实际情况记录到环境表里。

```bash
python - <<'PY'
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu available:", torch.npu.is_available())
PY
```

排查 `libhccl.so`：

```bash
find ~/Ascend/cann-9.1.0 -name 'libhccl.so*'
```

```bash
python - <<'PY'
import ctypes.util

print(ctypes.util.find_library("hccl"))
PY
```

安装 Transformers 后记录版本：

```bash
python - <<'PY'
import transformers

print(transformers.__version__)
PY
```
