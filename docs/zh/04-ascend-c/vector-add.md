# Vector Add 入门

本节使用 Ascend C 实现向量加法，了解 Host/Device 工程结构、tiling、数据搬运和 Vector API。

## 1. 工程目录

```text
src/04-ascend-c/vector_add/
```

| 文件 | 作用 |
|:---|:---|
| `add_custom_kernel.cpp` | Device 侧 CopyIn、Compute 和 CopyOut |
| `add_custom_host.cpp` | Host 侧 tiling 和算子注册 |
| `add_custom_tiling.h` | tiling 参数结构 |
| `CMakeLists.txt` | 编译配置 |

## 2. 加载 CANN

```bash
export CANN_SET_ENV=/path/to/cann/set_env.sh
source "$CANN_SET_ENV"
```

将 Toolkit 目录保存为 `ASCEND_INSTALL_PATH`：

```bash
export ASCEND_INSTALL_PATH=/path/to/ascend-toolkit/latest
```

## 3. 设置 SoC

查看设备信息：

```bash
npu-smi info
```

打开 `CMakeLists.txt`，将 `ASCEND_COMPUTE_UNIT` 设置为对应 SoC，例如：

```cmake
set(ASCEND_COMPUTE_UNIT ascend910b)
```

## 4. 编译

```bash
cd src/04-ascend-c/vector_add
export CMAKE_PREFIX_PATH="${ASCEND_INSTALL_PATH}/compiler/tikcpp/ascendc_kernel_cmake:${CMAKE_PREFIX_PATH}"
cmake -S . -B build
cmake --build build --target binary package -j
```

编译完成后，在 `build/` 中查看二进制和安装包。

## 5. 正确性对照

使用 PyTorch 生成参考结果：

```python
import torch
import torch_npu

x = torch.randn(8 * 2048, dtype=torch.float16, device="npu")
y = torch.randn(8 * 2048, dtype=torch.float16, device="npu")
z_ref = x + y
```

自定义算子输出与 `z_ref` 使用相同 shape 和 dtype 比较，并记录最大误差与平均误差。

## 6. 实验记录

| 项目 | 内容 |
|:---|:---|
| CANN 版本 |  |
| SoC |  |
| 输入 shape / dtype |  |
| 编译命令 |  |
| 产物路径 |  |
| 最大误差 |  |
| 平均误差 |  |
| 单次耗时 |  |
