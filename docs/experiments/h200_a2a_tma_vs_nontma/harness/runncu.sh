#!/bin/bash
# ncu profiling: single-sender (concurrent=0) so replay doesn't destroy contention semantics.
# Profiles rank0's kernel only; skips probe(1)+warmup(3) launches.
cd /data/prof_a2a
mkdir -p reports
set -x

PEER_METRICS="lts__t_requests_aperture_peer_op_write.sum,lts__t_sectors_aperture_peer_op_write.sum,lts__t_sectors_aperture_peer.sum,lts__t_sectors_aperture_device_op_write.sum,lts__t_sectors_aperture_device.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_srcunit_tex_op_write.sum,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_drain_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__issue_active.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum"

# 1) non-TMA faithful, repo-ish cfg (th512 un4 f12)
ncu --set full --launch-skip 4 --launch-count 1 -k "regex:a2a_copy_generic" \
    -o reports/nontma_full -f \
    ./a2a_harness bench --path nontma --mode 0 --ws 8 --N 32768 --threads 512 --unroll 4 --factor 12 --concurrent 0 --iters 3

# 2) non-TMA custom metrics (cheap, 1 pass-ish)
ncu --metrics $PEER_METRICS --launch-skip 4 --launch-count 1 -k "regex:a2a_copy_generic" \
    --csv ./a2a_harness bench --path nontma --mode 0 --ws 8 --N 32768 --threads 512 --unroll 4 --factor 12 --concurrent 0 --iters 3 \
    > reports/nontma_peer_metrics.csv 2>&1

# 3) non-TMA source counters (per-line stalls)
ncu --set full --section SourceCounters --import-source yes --launch-skip 4 --launch-count 1 -k "regex:a2a_copy_generic" \
    -o reports/nontma_src -f \
    ./a2a_harness bench --path nontma --mode 0 --ws 8 --N 32768 --threads 512 --unroll 4 --factor 12 --concurrent 0 --iters 3

# 4) TMA repo-default cfg (tn15 ts1 stg4 bdiv4)
ncu --set full --launch-skip 4 --launch-count 1 -k "regex:a2a_tma_kernel" \
    -o reports/tma_repo_full -f \
    ./a2a_harness bench --path tma --mode 0 --ws 8 --N 32768 --tile_n 15 --tile_s 1 --stages 4 --bdiv 4 --concurrent 0 --iters 3

# 5) TMA best cfg (tn16 ts4 stg8 bdiv2)
ncu --set full --launch-skip 4 --launch-count 1 -k "regex:a2a_tma_kernel" \
    -o reports/tma_best_full -f \
    ./a2a_harness bench --path tma --mode 0 --ws 8 --N 32768 --tile_n 16 --tile_s 4 --stages 8 --bdiv 2 --concurrent 0 --iters 3

# 6) TMA peer metrics for both cfgs
ncu --metrics $PEER_METRICS --launch-skip 4 --launch-count 1 -k "regex:a2a_tma_kernel" \
    --csv ./a2a_harness bench --path tma --mode 0 --ws 8 --N 32768 --tile_n 15 --tile_s 1 --stages 4 --bdiv 4 --concurrent 0 --iters 3 \
    > reports/tma_repo_peer_metrics.csv 2>&1
ncu --metrics $PEER_METRICS --launch-skip 4 --launch-count 1 -k "regex:a2a_tma_kernel" \
    --csv ./a2a_harness bench --path tma --mode 0 --ws 8 --N 32768 --tile_n 16 --tile_s 4 --stages 8 --bdiv 2 --concurrent 0 --iters 3 \
    > reports/tma_best_peer_metrics.csv 2>&1

# 7) non-TMA write-only variant metrics (read-side removed)
ncu --metrics $PEER_METRICS --launch-skip 4 --launch-count 1 -k "regex:a2a_copy_generic" \
    --csv ./a2a_harness bench --path nontma --mode 0 --ws 8 --N 32768 --threads 512 --unroll 4 --factor 12 --concurrent 0 --iters 3 --variant 1 \
    > reports/nontma_wo_peer_metrics.csv 2>&1

ls -la reports/
