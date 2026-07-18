# 环境检查清单

## 测试环境

| 项目 | 版本或说明 |
|:---|:---|
| 硬件 | IT22HMDA_4_S（2 芯片），64 GB HBM/芯片，共 128 GB |
| 系统 | Ubuntu 20.04.5 LTS，kernel 5.10.0-182（aarch64） |
| 驱动 | 25.5.5（Innerversion: V100R001C23SPC009B220） |
| 固件 | `npu-smi info -t board` 返回 `NA` |
| CANN | 9.0.0（`~/Ascend/cann-9.0.0/`） |
| Python | 3.11.4 |
| PyTorch | torch 2.7.1+cpu |
| torch_npu | 2.7.1.post2.dev20251226 |
| Transformers | 未安装 |

## 检查项

- [x] `npu-smi info` 能看到 NPU。
- [x] CANN Toolkit 路径存在。
- [x] CANN 环境变量已加载。
- [x] `libhccl.so` 可以被动态链接器找到。
- [x] Python 环境可用。
- [x] PyTorch 可导入。
- [x] `torch_npu` 可导入。
- [x] 最小 NPU 张量计算能运行。

Transformers 不属于本章的通过条件，在 01 Inference 章安装。

## 常用命令

```bash
npu-smi info
```

查看板卡、驱动和固件信息。`7` 是本次实验中的 NPU ID，其他机器按 `npu-smi info` 第一列替换：

```bash
npu-smi info -t board -i 7
sed -n '1,80p' /usr/local/Ascend/driver/version.info
```

查找 CANN 环境脚本：

```bash
find /usr/local/Ascend ~/Ascend -name set_env.sh 2>/dev/null
```

如果 `/usr/local/Ascend` 下面只有 `driver`，说明当前路径里只能看到驱动，不代表 toolkit 一定装在这里。继续检查 `~/Ascend`、镜像文档或管理员提供的 CANN 安装路径。

本次环境返回四个脚本，其中课程使用的是 CANN 根目录脚本：

```text
~/Ascend/cann-9.0.0/set_env.sh
```

`tools/msmemscope` 和 `ascendnpu-ir` 下面的脚本只服务于对应工具。`~/Ascend/ascend-toolkit/set_env.sh` 是指向 CANN 根目录脚本的符号链接，可以这样确认：

```bash
readlink -f ~/Ascend/ascend-toolkit/set_env.sh
```

按实际路径加载主脚本：

```bash
source ~/Ascend/ascend-toolkit/set_env.sh
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

不要使用 `atc --version` 判断本机 CANN 版本，本次环境中的 ATC 不支持这个参数。安装版本记录在 `ascend_toolkit_install.info`：

```bash
grep -E '^(version|innerversion|arch|os|path)=' \
  "${ASCEND_HOME_PATH}/aarch64-linux/ascend_toolkit_install.info"
```

排查 `libhccl.so`：

```bash
find "${ASCEND_HOME_PATH}" -name 'libhccl.so*'
```

```bash
python - <<'PY'
import ctypes.util

print(ctypes.util.find_library("hccl"))
PY
```

本次实测输出为 `libhccl.so`，说明动态链接器可以找到 HCCL。

仓库提供了完整检查脚本：

```bash
bash src/00_environment/check_environment.sh
```
