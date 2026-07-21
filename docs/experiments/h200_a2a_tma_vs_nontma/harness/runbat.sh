#!/bin/bash
# Isolation battery: pair P2P BW per write-method + wire-level NVLink counters + a2a variants.
cd /data/prof_a2a

snap_d() { nvidia-smi nvlink -gt d -i 0 | awk '/Data Tx/{s+=$5} END{print s}'; }
snap_r() { nvidia-smi nvlink -gt r -i 0 | awk '/Raw Tx/{s+=$5} END{print s}'; }
run_counted() { # label cmd...
  local label="$1"; shift
  local d0=$(snap_d) r0=$(snap_r)
  "$@"
  local d1=$(snap_d) r1=$(snap_r)
  echo "WIRE $label dataTx_KiB=$((d1-d0)) rawTx_KiB=$((r1-r0))"
}

echo "===== PAIR: copy-engine DMA reference (512MB GPU0->GPU1) ====="
./a2a_harness pair --path memcpy --mb 512 --iters 20
./a2a_harness pair --path memcpy --mb 512 --iters 20 --local 1

echo "===== PAIR: nontma (SM st.global 16B) GPU0->GPU1, threads x factor ====="
for th in 128 256 512 1024; do
  for f in 4 8 16 32; do
    ./a2a_harness pair --path nontma --mb 512 --threads $th --unroll 4 --factor $f --iters 10
  done
done
echo "--- local-dst reference (same kernel, dst on GPU0) ---"
./a2a_harness pair --path nontma --mb 512 --threads 512 --factor 16 --iters 10 --local 1

echo "===== PAIR: tma bulk tensor GPU0->GPU1, tile_n x tile_s (tile_bytes = tn*ts*256B) ====="
for tn in 8 16 32 64 128; do
  for ts in 1 2 4; do
    ./a2a_harness pair --path tma --mb 512 --tile_n $tn --tile_s $ts --iters 10
  done
done
echo "--- local-dst reference ---"
./a2a_harness pair --path tma --mb 512 --tile_n 64 --tile_s 2 --iters 10 --local 1

echo "===== WIRE: data/raw NVLink bytes per method (single sender) ====="
run_counted pair_memcpy  ./a2a_harness pair --path memcpy --mb 512 --iters 20
run_counted pair_nontma  ./a2a_harness pair --path nontma --mb 512 --threads 512 --factor 16 --iters 20
run_counted pair_tma     ./a2a_harness pair --path tma --mb 512 --tile_n 16 --tile_s 4 --iters 20

echo "===== A2A variants (mode0 ws=8 N=32768 H=128) ====="
CFG_NT="--threads 512 --unroll 4 --factor 12"
CFG_TMA="--tile_n 15 --tile_s 1 --stages 4 --bdiv 4"
echo "--- nontma faithful, concurrent ---"
./a2a_harness bench --path nontma --mode 0 --ws 8 --N 32768 $CFG_NT --concurrent 1 --iters 20
echo "--- nontma write-only (no src reads), concurrent ---"
./a2a_harness bench --path nontma --mode 0 --ws 8 --N 32768 $CFG_NT --concurrent 1 --iters 20 --variant 1
echo "--- nontma local-dst (no NVLink), single ---"
./a2a_harness bench --path nontma --mode 0 --ws 8 --N 32768 $CFG_NT --concurrent 0 --iters 20 --localdst 1
echo "--- nontma faithful, single sender (no contention) ---"
./a2a_harness bench --path nontma --mode 0 --ws 8 --N 32768 $CFG_NT --concurrent 0 --iters 20
echo "--- tma faithful (repo default cfg), concurrent ---"
./a2a_harness bench --path tma --mode 0 --ws 8 --N 32768 $CFG_TMA --concurrent 1 --iters 20
echo "--- tma BEST cfg (tn=16 ts=4 stg=8 bdiv=2), concurrent ---"
CFG_TMAB="--tile_n 16 --tile_s 4 --stages 8 --bdiv 2"
./a2a_harness bench --path tma --mode 0 --ws 8 --N 32768 $CFG_TMAB --concurrent 1 --iters 20
echo "--- tma local-dst, single ---"
./a2a_harness bench --path tma --mode 0 --ws 8 --N 32768 $CFG_TMA --concurrent 0 --iters 20 --localdst 1
echo "--- tma faithful, single sender ---"
./a2a_harness bench --path tma --mode 0 --ws 8 --N 32768 $CFG_TMA --concurrent 0 --iters 20

echo "===== WIRE: a2a single-sender data/raw per path ====="
run_counted a2a_nontma_single ./a2a_harness bench --path nontma --mode 0 --ws 8 --N 32768 $CFG_NT --concurrent 0 --iters 50
run_counted a2a_tma_single    ./a2a_harness bench --path tma --mode 0 --ws 8 --N 32768 $CFG_TMA --concurrent 0 --iters 50
run_counted a2a_tmabest_single ./a2a_harness bench --path tma --mode 0 --ws 8 --N 32768 $CFG_TMAB --concurrent 0 --iters 50
echo "--- tma BEST single sender (timing) ---"
./a2a_harness bench --path tma --mode 0 --ws 8 --N 32768 $CFG_TMAB --concurrent 0 --iters 20
echo "payload/iter(single sender egress) = numel*2*7/8 = 117.44 MB; iters incl 3 warmup + 1 probe"
