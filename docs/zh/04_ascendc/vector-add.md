# Vector Add 入门

## 本节目标

完成一个基础 Ascend C Vector Add 算子，跑通编译、执行、正确性检查和 profiling。

第一轮先做两件事：

1. 确认当前服务器能编译 Ascend C 算子工程。
2. 记录编译路径、SoC 名称、产物路径和遇到的问题。

正确性测试和 PyTorch 接入放在下一步，不要在第一次编译时同时处理太多变量。

## 工程目录

本节工程在：

```text
src/04_ascendc/vector_add/
```

主要文件：

| 文件 | 作用 |
|:---|:---|
| `add_custom_kernel.cpp` | device 侧计算逻辑，展示 CopyIn / Compute / CopyOut |
| `add_custom_host.cpp` | host 侧 tiling 和算子注册 |
| `add_custom_tiling.h` | tiling 参数结构 |
| `CMakeLists.txt` | 编译和打包配置 |
| `README.md` | 工程说明 |

## 检查 CANN 路径

先确认 CANN Toolkit 实际安装在哪里：

```bash
find /usr/local/Ascend ~/Ascend -name set_env.sh 2>/dev/null
```

如果找到了 `set_env.sh`，按实际路径加载：

```bash
source /path/to/set_env.sh
```

如果服务器使用的是预置环境，记录实际的 `ASCEND_HOME`、`ASCEND_INSTALL_PATH`、`PATH` 和 `LD_LIBRARY_PATH`。不要把不存在的固定路径写进教程。

## 确认 SoC 名称

当前工程默认：

```cmake
set(ASCEND_COMPUTE_UNIT ascend910b)
```

第一轮需要用服务器实际信息确认这个值是否正确。可以先记录：

```bash
npu-smi info
```

如果编译时报 SoC 或 product name 相关错误，先不要改其他代码，只调整 `ASCEND_COMPUTE_UNIT` 并记录结果。

## 编译

进入工程目录：

```bash
cd src/04_ascendc/vector_add
```

设置 CMake Module 搜索路径。下面路径只是示例，实际路径以服务器为准：

```bash
export ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
export CMAKE_PREFIX_PATH=${ASCEND_INSTALL_PATH}/compiler/tikcpp/ascendc_kernel_cmake:$CMAKE_PREFIX_PATH
```

编译：

```bash
rm -rf build
mkdir build
cd build
cmake ..
make -j binary package
```

如果成功，记录产物：

```text
build/<vendor_name>_run.tar.gz
```

如果失败，记录：

| 项目 | 内容 |
|:---|:---|
| CANN 路径 |  |
| SoC 名称 |  |
| cmake 命令 |  |
| 报错关键行 |  |
| 下一步处理 |  |

## 正确性验证

编译成功后再进入正确性验证。第一轮先用 PyTorch 参考实现作为对照：

```python
import torch
import torch_npu

x = torch.randn(8 * 2048, dtype=torch.float16, device="npu")
y = torch.randn(8 * 2048, dtype=torch.float16, device="npu")
z_ref = x + y
```

自定义算子安装和调用方式需要结合本机 CANN 9.1.0 的实际产物确认。确认前，正文只记录编译结果，不写成已经完成模型接入。

## 记录

把本次结果写到案例报告或本节末尾，至少包含：

- CANN 路径。
- SoC 名称。
- 编译命令。
- 编译是否成功。
- 产物路径。
- 如果失败，贴关键报错。
- 下一步准备做正确性测试还是先调整工程模板。
