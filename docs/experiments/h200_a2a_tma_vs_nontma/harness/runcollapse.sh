#!/bin/bash
# Characterize the concurrent-only collapse: resident-blocks knob (smempad), nontma factor cliff,
# repo TMA candidates at N=16384, and the real tuner picks (verbose lib).
cd /data/prof_a2a

echo "===== A. resident-block throttle via smempad (tn16 ts4 stg2 bdiv2, m0 ws8 N=32768, concurrent) ====="
echo "  base smem 32.4KB: pad 0->7blk/SM, 16K->4, 32K->3, 64K->2, 96K/160K->1"
for pad in 0 16384 32768 65536 98304 163840; do
  ./a2a_harness bench --path tma --mode 0 --ws 8 --N 32768 --tile_n 16 --tile_s 4 --stages 2 --bdiv 2 --smempad $pad --concurrent 1 --iters 15
done
echo "===== A2. same config, single sender (collapse should vanish) ====="
for pad in 0 163840; do
  ./a2a_harness bench --path tma --mode 0 --ws 8 --N 32768 --tile_n 16 --tile_s 4 --stages 2 --bdiv 2 --smempad $pad --concurrent 0 --iters 15
done

echo "===== B. nontma factor cliff th512 un4 (concurrent) ====="
for f in 4 8 12 16 20 24 28 32; do
  ./a2a_harness bench --path nontma --mode 0 --ws 8 --N 32768 --threads 512 --unroll 4 --factor $f --concurrent 1 --iters 15
done
echo "===== B2. nontma factor curve, single sender (cliff should vanish) ====="
for f in 8 16 32; do
  ./a2a_harness bench --path nontma --mode 0 --ws 8 --N 32768 --threads 512 --unroll 4 --factor $f --concurrent 0 --iters 15
done

echo "===== C. repo TMA candidate set at N=16384 (bench measured 167 GB/s) concurrent ====="
for c in "15 1" "8 1" "16 1"; do
  set -- $c
  ./a2a_harness bench --path tma --mode 0 --ws 8 --N 16384 --tile_n $1 --tile_s $2 --stages 4 --bdiv 4 --concurrent 1 --iters 15
done
./a2a_harness bench --path tma --mode 0 --ws 8 --N 16384 --tile_n 16 --tile_s 4 --stages 8 --bdiv 2 --concurrent 1 --iters 15
./a2a_harness bench --path nontma --mode 0 --ws 8 --N 16384 --threads 512 --unroll 4 --factor 12 --concurrent 1 --iters 15

echo "===== D. pair local-dst HBM references (fixing earlier --local typo) ====="
./a2a_harness pair --path nontma --mb 512 --threads 512 --factor 16 --iters 10 --localdst 1
./a2a_harness pair --path tma --mb 512 --tile_n 64 --tile_s 2 --iters 10 --localdst 1

echo "===== E. real tuner picks (rebuilt lib, FAST_ULYSSES_TUNE_VERBOSE=1) ====="
cd /data/fast-ulysses && source /data/.torch/bin/activate
echo "--- forced TMA, N=16384,32768 ---"
PROF_N=16384,32768 PROF_MODE=0 CUSTOM_ULYSSES_USE_TMA=1 FAST_ULYSSES_TUNE_VERBOSE=1 \
  torchrun --nproc_per_node=8 benchmark/bench_uniform.py 2>&1 | grep -E "tma-at|N=" | sort | uniq -c | sort -rn | head -40
echo "--- auto path, N=32768 ---"
PROF_N=32768 PROF_MODE=0 FAST_ULYSSES_TUNE_VERBOSE=1 \
  torchrun --nproc_per_node=8 benchmark/bench_uniform.py 2>&1 | grep -E "at-pick|nontma-at|N=" | head -30
