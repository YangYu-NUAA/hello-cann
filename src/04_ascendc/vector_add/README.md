# Vector Add 算子工程

hello-cann 04 Ascend C 章第一个算子。基于 Ascend 官方 AddCustomTiny 模板（CANN 8.3+），加入教学注释。

## 工程结构

```text
vector_add/
├── add_custom_kernel.cpp   # kernel 侧：device 上跑的计算逻辑（三段式 CopyIn/Compute/CopyOut）
├── add_custom_host.cpp     # host 侧：tiling 计算 + 算子注册（OpDef + OP_ADD）
├── add_custom_tiling.h     # tiling 参数定义（host 传给 kernel 的切分指令）
├── CMakeLists.txt          # 编译工程（生成可安装的算子包）
└── README.md               # 本文件
```

## 编译

前置条件：CANN 8.3 及以上（课程环境 CANN 9.0.0 满足）。

先找到 CANN Toolkit 的实际路径：

```bash
find /usr/local/Ascend ~/Ascend -name set_env.sh 2>/dev/null
```

加载环境后再编译。下面的 `ASCEND_INSTALL_PATH` 只是示例，服务器上以实际路径为准：

```bash
export ASCEND_INSTALL_PATH=/path/to/ascend-toolkit/latest
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
export CMAKE_PREFIX_PATH=${ASCEND_INSTALL_PATH}/compiler/tikcpp/ascendc_kernel_cmake:$CMAKE_PREFIX_PATH

cd src/04_ascendc/vector_add
rm -rf build && mkdir build && cd build
cmake .. && make -j binary package
```

编译产物是 `build/<vendor_name>_run.tar.gz`。第一轮先记录是否编译成功、产物路径和关键报错；安装和 Python 侧调用在下一步补齐。

## 调用验证

编译成功后，用 PyTorch 参考实现准备正确性对比：

```python
import torch
import torch_npu
x = torch.randn(8 * 2048, dtype=torch.float16, device="npu")
y = torch.randn(8 * 2048, dtype=torch.float16, device="npu")
z_ref = x + y
```

自定义算子的安装、aclnn 调用和 PyTorch 侧封装，需要结合 CANN 9.0.0 实测产物补到：
- `docs/zh/04_ascendc/vector-add.md`（本算子教程）
- `docs/zh/04_ascendc/model-integration.md`（接入 PyTorch）

## 教学要点

| 概念 | 在哪个文件体现 |
|:---|:---|
| 三段式编程范式（CopyIn/Compute/CopyOut） | `add_custom_kernel.cpp` 的 `Process()` |
| 内存层级（Global <-> Local） | `add_custom_kernel.cpp` 的 `DataCopy` |
| 多核切分 | `add_custom_kernel.cpp` 的 `blockLength = totalLength / GetBlockNum()` |
| 双缓冲流水 | `add_custom_kernel.cpp` 的 `BUFFER_NUM = 2` |
| Tiling 机制 | `add_custom_host.cpp` 的 `TilingFunc` + `add_custom_tiling.h` |
| 算子注册 | `add_custom_host.cpp` 的 `OpDef` + `OP_ADD` |

## 参考来源

- Ascend 官方 AddCustomTiny 样例：https://gitee.com/ascend/samples/operator/ascendc/0_introduction/1_add_frameworklaunch/AddCustomTiny
- Ascend C 算子开发文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/devguide/opdevc/atlopdevc
