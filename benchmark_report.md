# fast-ulysses benchmark report

- world_size: 8
- dtype: bfloat16
- backend: mlx5
- NCCL_P2P_LEVEL: <unset>
- warmup: 5 calls/case
- measurement: 1 call(s)/trial, 20 trials, slowest rank then median

All bandwidths are decimal GB/s. `bus` counts only bytes sent to remote ranks; `aggregate` is the sum across all ranks.

| Shape | Mode | Raw NCCL ms | NCCL alg GB/s | NCCL bus GB/s | NCCL aggregate GB/s | NCCL + layout ms | Layout GB/s | Fast ms | Fast GB/s | Raw / fast | Layout / fast |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192,8,128 | 0 | 0.093 | 22.50 | 19.69 | 157.49 | 0.100 | 18.41 | 0.092 | 20.03 | 1.02x | 1.09x |
| 8192,8,128 | 1 | 0.092 | 22.72 | 19.88 | 159.05 | 0.103 | 17.89 | 0.093 | 19.81 | 1.00x | 1.11x |
| 32760,40,128 | 0 | 1.167 | 35.93 | 31.44 | 251.51 | 1.296 | 28.32 | 0.933 | 39.34 | 1.25x | 1.39x |
| 32760,40,128 | 1 | 1.168 | 35.91 | 31.42 | 251.34 | 1.267 | 28.96 | 0.948 | 38.68 | 1.23x | 1.34x |
| 37824,56,128 | 0 | 1.795 | 37.75 | 33.04 | 264.28 | 2.069 | 28.66 | 1.486 | 39.91 | 1.21x | 1.39x |
| 37824,56,128 | 1 | 1.797 | 37.71 | 33.00 | 263.96 | 2.048 | 28.95 | 1.509 | 39.29 | 1.19x | 1.36x |
| 75600,40,128 | 0 | 2.549 | 37.97 | 33.22 | 265.78 | 2.951 | 28.69 | 2.125 | 39.84 | 1.20x | 1.39x |
| 75600,40,128 | 1 | 2.550 | 37.95 | 33.20 | 265.63 | 2.912 | 29.07 | 2.142 | 39.54 | 1.19x | 1.36x |
