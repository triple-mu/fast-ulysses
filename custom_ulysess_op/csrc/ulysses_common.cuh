#pragma once
#include <cstdint>
#include <vector>

namespace ulysess {

// host/device 共享的维度描述（int32 字段，kernel 内用 int64 做地址运算）。
struct Ulysses4DDims {
    int32_t b, s_local, s_global, n_local, n_global, d, rank;
};

template<int WS>
struct PeerPtrs {
    void* p[WS];
};

// baseline 为按字节直拷，Identity 不做任何变换；未来 RoPE/RMSNorm 特化此 hook。
struct EpilogueIdentity {
    __device__ __forceinline__ void operator()(uint8_t* /*row_bytes*/, int /*row_off*/) const {}
};

enum class KernelArch {
    kPlain80,
    kPlain90,
    kPlain100
};

// 变长（uneven s/n，可不被 world_size 整除）：前缀偏移数组（world_size ≤ 8 → ≤ 9 项）。
// 均匀情形是其特例（s_off[r]=r*s_local、n_off[r]=r*n_local）。调用方提供 split（无运行时 gather）。
struct SplitInfo {
    int32_t world_size, rank, b, d;
    int32_t s_off[9];  // s_off[r]=rank r 序列起点；s_off[world_size]=S（总序列）
    int32_t n_off[9];  // n_off[r]=rank r 头起点；n_off[world_size]=N（总头数）
};

// 自由函数：纯 CUDA 向量化直写 peer 对称显存（host 侧已用 nvshmem_ptr 取得
// peer_ptrs）。
void launch_a2a(const void*                  src,
                const std::vector<uint64_t>& peer_ptrs,
                const Ulysses4DDims&         dims,
                int                          mode,
                int                          elem_size,
                cudaStream_t                 stream);

// 变长版本：源端路由（遍历本地输入，每个头/序列位按 split 偏移找归属 peer 写入）。
void launch_a2a_varlen(const void*                  src,
                       const std::vector<uint64_t>& peer_ptrs,
                       const SplitInfo&             sp,
                       int                          mode,
                       int                          elem_size,
                       cudaStream_t                 stream);

}  // namespace ulysess
