/**
 * @file add_custom_tiling.h
 * @brief Vector Add 的 tiling 参数定义
 *
 * Tiling 参数是 host 侧传给 kernel 侧的「切分指令」。
 * kernel 侧通过 GET_TILING_DATA 拿到这个结构体，据此做多核切分和流水。
 */
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H
#include <cstdint>

struct AddCustomTilingData {
    uint32_t totalLength;  // 总元素数，kernel 侧据此切分到各核
    uint32_t tileNum;      // 每核数据切成几块，用于流水
};

#endif // ADD_CUSTOM_TILING_H
