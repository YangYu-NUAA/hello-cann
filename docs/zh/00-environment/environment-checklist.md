# 环境检查

本节确认 NPU、CANN、Python 和 `torch_npu` 已经可用。

## 1. 查看 NPU

```bash
npu-smi info
```

记录输出中的 NPU 型号、设备编号、显存和驱动版本。需要查看单个设备时，将 `NPU_ID` 设为 `npu-smi info` 第一列中的编号：

```bash
export NPU_ID=0
npu-smi info -t board -i "$NPU_ID"
```

## 2. 加载 CANN 环境

在常见安装目录中查找 CANN 环境脚本：

```bash
find /usr/local/Ascend "$HOME/Ascend" -name set_env.sh 2>/dev/null
```

CANN 根目录或 `ascend-toolkit` 目录中的 `set_env.sh` 用于加载完整开发环境。先把实际路径保存到变量，再加载：

```bash
export CANN_SET_ENV=/path/to/cann/set_env.sh
source "$CANN_SET_ENV"
```

云镜像可能已经自动加载 CANN。此时 `ASCEND_HOME_PATH`、`PATH` 和 `LD_LIBRARY_PATH` 中已经包含 Ascend 目录，可以直接继续检查：

```bash
printf 'ASCEND_HOME_PATH=%s\n' "$ASCEND_HOME_PATH"
```

查看 CANN 安装信息：

```bash
find "${ASCEND_HOME_PATH:-$HOME/Ascend}" -name ascend_toolkit_install.info -print -quit
```

## 3. 检查 Python 环境

```bash
python -c 'import platform, torch, torch_npu; print("Python:", platform.python_version()); print("PyTorch:", torch.__version__); print("torch_npu:", torch_npu.__version__); print("NPU available:", torch.npu.is_available()); print("NPU count:", torch.npu.device_count())'
```

输出中的 `NPU available` 应为 `True`。

## 4. 检查 HCCL 动态库

```bash
find "${ASCEND_HOME_PATH}" -name 'libhccl.so*'
```

也可以让 Python 查询动态链接器：

```bash
python -c 'import ctypes.util; print(ctypes.util.find_library("hccl"))'
```

## 5. 运行仓库检查脚本

```bash
bash src/00-environment/check_environment.sh
```

脚本会输出环境变量、包版本和 NPU 可用状态。设备和软件版本可按 [版本矩阵](../06-references/version-matrix.md) 的格式记录。

## 检查结果

- [ ] `npu-smi info` 能列出 NPU。
- [ ] CANN 环境已经加载。
- [ ] Python 可以导入 PyTorch 和 `torch_npu`。
- [ ] `torch.npu.is_available()` 返回 `True`。
- [ ] HCCL 动态库可以被找到。

下一节：[`torch_npu` 最小校验](torch-npu-check.md)。
