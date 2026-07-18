/**
 * @file add_custom_kernel.cpp
 * @brief hello-cann 04 Ascend C 第一个算子：Vector Add (kernel 侧)
 *
 * 工程结构对齐 Ascend 官方 AddCustomTiny 样例（CANN 8.3+）：
 *   https://gitee.com/ascend/samples/operator/ascendc/0_introduction/1_add_frameworklaunch/AddCustomTiny
 *
 * 教学要点（对应 docs/zh/04_ascendc/vector-add.md）：
 *   1. 三段式编程范式：CopyIn -> Compute -> CopyOut
 *   2. 内存层级：Global Memory (GM) <-> Local Memory (LM，经 TQue 管理)
 *   3. 多核切分：blockLength = totalLength / BlockNum，每核处理一段
 *   4. tiling 切分：每核数据再切成 tileNum * BUFFER_NUM 块，做流水
 *
 * 这份实现不采用“一线程一元素”的入门写法，而是用 TQue/DataCopy/Add
 * 展示 Ascend C 里的数据搬运和片上存储流水。
 */
#include "kernel_operator.h"
#include "add_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // 双缓冲，让搬运和计算重叠（流水）

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        // 多核切分：每个 block 处理 totalLength / BlockNum 个元素
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // 为当前 block 设置 GM 偏移：每个 block 看到自己负责的那一段
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        // 在 Local Memory 上分配双缓冲队列（搬运/计算并行）
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);    // GM -> LM
            Compute(i);   // LM 上做 Add
            CopyOut(i);   // LM -> GM
        }
    }

private:
    // 第 1 步：把 x、y 从 Global Memory 搬到 Local Memory
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    // 第 2 步：在 Local Memory 上做加法（Vector API 只能操作 LocalTensor）
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    // 第 3 步：把结果从 Local Memory 搬回 Global Memory
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

// 核函数入口：host 侧通过 aclnn 调用到这里
// 注意 extern "C" + __global__ + __aicore__ 三个修饰符是固定的
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelAdd op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
