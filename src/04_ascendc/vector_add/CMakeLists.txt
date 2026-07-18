# hello-cann 04 Ascend C: Vector Add 算子工程
# 基于 Ascend 官方 AddCustomTiny 模板（CANN 8.3+）
# 文档：docs/zh/04_ascendc/vector-add.md
cmake_minimum_required(VERSION 3.16.0)
project(opp)

# 目标 SoC：第一轮先用 ascend910b 试编译，服务器实测后再确认。
# 如需支持其他型号，可改为 ascend910 / ascend310p / ascend310b 等。
set(ASCEND_COMPUTE_UNIT ascend910b)

# vendor_name 是算子包的命名空间，编译产物会以 <vendor_name> 为前缀
# 例：custom_opapi / custom_optiling / ascendc_kernels
if(NOT DEFINED vendor_name)
    set(vendor_name "hello_cann")
endif()

find_package(ASC REQUIRED)

# 1. 声明算子包
npu_op_package(${vendor_name}
    TYPE RUN
)

# 2. 从 host 源码生成 aclnn 自动代码（注册、调度等）
file(GLOB host_ops_srcs ${CMAKE_CURRENT_SOURCE_DIR}/add_custom_host.cpp)
npu_op_code_gen(
    SRC ${host_ops_srcs}
    PACKAGE ${vendor_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    OPTIONS
    OPS_PRODUCT_NAME ${ASCEND_COMPUTE_UNIT}
)

# 3. 编译 aclnn 库（host 侧调用入口）
file(GLOB autogen_aclnn_srcs ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
set_source_files_properties(${autogen_aclnn_srcs} PROPERTIES GENERATED TRUE)
npu_op_library(cust_opapi ACLNN
    ${autogen_aclnn_srcs}
)

# 4. 编译 tiling 库（host 侧 tiling 逻辑）
npu_op_library(cust_optiling TILING
    ${host_ops_srcs}
)

# 5. 编译 kernel 库（device 侧计算）
npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_SOURCE_DIR}/
    TILING_LIBRARY cust_optiling
)

npu_op_kernel_sources(ascendc_kernels
    OP_TYPE AddCustom
    KERNEL_FILE add_custom_kernel.cpp
)

# 6. 打包成可安装的算子包
npu_op_package_add(${vendor_name}
    LIBRARY
    cust_opapi
    cust_optiling
    ascendc_kernels
)
