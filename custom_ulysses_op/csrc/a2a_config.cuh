#pragma once
#include "ulysses_common.cuh"
#include <algorithm>
#include <tuple>

namespace ulysses {

// A2A launch config: TMA path uses tile_n/tile_s/stages/bdiv; non-TMA path uses threads/unroll/blocks.
// blocks=0 means auto-selected by pick_blocks.
struct A2AConfig {
    int tile_n = 0, tile_s = 1, stages = 4, bdiv = 4;
    int threads = 512, unroll = 4, blocks = 0;
};

// Defaults per mode (near-optimal for DiT):
// mode0 -> tile_n=max(1,n_local-1), tile_s=1 (non-divisor small tile_n yields more n-tiles, higher TMA concurrency);
// mode1 -> tile_n=n_local, tile_s=2. Both use stages=4, bdiv=4, threads=512, unroll=4, blocks=0.
inline A2AConfig default_config(int mode, int n_local)
{
    A2AConfig c;
    if (mode == 0) {
        c.tile_n = std::max(1, n_local - 1);
        c.tile_s = 1;
    }
    else {
        c.tile_n = std::max(1, n_local);
        c.tile_s = 2;
    }
    c.stages  = 4;
    c.bdiv    = 4;
    c.threads = 512;
    c.unroll  = 4;
    c.blocks  = 0;
    return c;
}

// autotune/cache key (excludes b and elem -- 2B path only): ws, mode, tma(0/1), n_local, s_local, d.
// The tma bit separates TMA / non-TMA configs (same shape differs per path, so caches must be split).
using ConfigKey = std::tuple<int, int, int, int, int, int>;

inline ConfigKey config_key(int ws, int mode, bool tma, const Ulysses4DDims& dims)
{
    return ConfigKey{ws, mode, tma ? 1 : 0, dims.n_local, dims.s_local, dims.d};
}

// Single source of truth for path selection (tri-state int: -1 auto / 0 off / 1 on). Both bindings'
// decide_tma and group::tune call it to avoid logic drift. auto: uniform on sm90+ -> TMA; varlen always
// non-TMA (TMA-varlen is slower). The sm<9 error for explicit use_tma=True is enforced by the caller's
// TORCH_CHECK; this function does not validate.
inline bool decide_tma_path(int use_tma_i, int sm_major, bool is_varlen)
{
    if (use_tma_i > 0)
        return true;
    if (use_tma_i == 0)
        return false;
    return sm_major >= 9 && !is_varlen;
}

}  // namespace ulysses
