/**
 * @file add_custom_host.cpp
 * @brief hello-cann 04 Ascend C 第一个算子：Vector Add (host 侧)
 *
 * host 侧做两件事：
 *   1. TilingFunc：根据输入 shape 计算 tiling 参数（totalLength / tileNum / BlockDim）
 *   2. OpDef：向 GE 注册算子的输入输出、dtype、支持的 SoC
 *
 * 对应 docs/zh/04_ascendc/vector-add.md 第 4 节「编译与运行」。
 */
#include "add_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace optiling {
// 这两个值是教学用的固定切分，实际工程会按 shape/SoC 动态计算
// BLOCK_DIM=8 表示用 8 个核并行；TILE_NUM=8 表示每核数据切成 8 块做流水
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    AddCustomTilingData *tiling = context->GetTilingData<AddCustomTilingData>();
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling->totalLength = totalLength;
    tiling->tileNum = TILE_NUM;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling


namespace ops {
class AddCustom : public OpDef {
public:
    explicit AddCustom(const char *name) : OpDef(name)
    {
        // 输入 x：必填，float16，ND 格式
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        // 输入 y：必填，float16，ND 格式
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        // 输出 z：必填，float16，ND 格式
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // 注册 tiling 函数和支持的 SoC
        // hello-cann 第一组环境是 Atlas A2（Ascend910B），优先支持这个
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(AddCustom);
} // namespace ops
