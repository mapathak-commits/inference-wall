# llama.cpp CPU inference bench — 2026-08-31T22:51:33Z


## Qwen_Qwen3.5-4B-bf16.gguf
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35 4B BF16                 |   8.06 GiB |     4.33 B | CPU        |       1 |           pp512 |          7.10 ± 0.01 |
| qwen35 4B BF16                 |   8.06 GiB |     4.33 B | CPU        |       1 |           tg128 |          1.60 ± 0.00 |
| qwen35 4B BF16                 |   8.06 GiB |     4.33 B | CPU        |       2 |           pp512 |         13.70 ± 0.35 |
| qwen35 4B BF16                 |   8.06 GiB |     4.33 B | CPU        |       2 |           tg128 |          2.84 ± 0.02 |
| qwen35 4B BF16                 |   8.06 GiB |     4.33 B | CPU        |       4 |           pp512 |         26.07 ± 0.18 |
| qwen35 4B BF16                 |   8.06 GiB |     4.33 B | CPU        |       4 |           tg128 |          3.88 ± 0.03 |
| qwen35 4B BF16                 |   8.06 GiB |     4.33 B | CPU        |       8 |           pp512 |         39.84 ± 0.47 |
| qwen35 4B BF16                 |   8.06 GiB |     4.33 B | CPU        |       8 |           tg128 |          3.92 ± 0.04 |
| qwen35 4B BF16                 |   8.06 GiB |     4.33 B | CPU        |      16 |           pp512 |         30.38 ± 1.12 |
| qwen35 4B BF16                 |   8.06 GiB |     4.33 B | CPU        |      16 |           tg128 |          0.21 ± 0.00 |

build: 458681e (1)

## Qwen_Qwen3.5-4B-Q8_0.gguf
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35 4B Q8_0                 |   4.29 GiB |     4.33 B | CPU        |       1 |           pp512 |          7.61 ± 0.06 |
| qwen35 4B Q8_0                 |   4.29 GiB |     4.33 B | CPU        |       1 |           tg128 |          2.73 ± 0.03 |
| qwen35 4B Q8_0                 |   4.29 GiB |     4.33 B | CPU        |       2 |           pp512 |         14.69 ± 0.42 |
| qwen35 4B Q8_0                 |   4.29 GiB |     4.33 B | CPU        |       2 |           tg128 |          4.71 ± 0.03 |
| qwen35 4B Q8_0                 |   4.29 GiB |     4.33 B | CPU        |       4 |           pp512 |         26.92 ± 0.26 |
| qwen35 4B Q8_0                 |   4.29 GiB |     4.33 B | CPU        |       4 |           tg128 |          6.83 ± 0.07 |
| qwen35 4B Q8_0                 |   4.29 GiB |     4.33 B | CPU        |       8 |           pp512 |         31.54 ± 0.73 |
| qwen35 4B Q8_0                 |   4.29 GiB |     4.33 B | CPU        |       8 |           tg128 |          6.88 ± 0.08 |
| qwen35 4B Q8_0                 |   4.29 GiB |     4.33 B | CPU        |      16 |           pp512 |         29.43 ± 1.03 |

## Qwen_Qwen3.5-4B-Q4_K_M.gguf
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | CPU        |       1 |           pp512 |          8.17 ± 0.02 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | CPU        |       1 |           tg128 |          3.63 ± 0.02 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | CPU        |       2 |           pp512 |         15.52 ± 0.72 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | CPU        |       2 |           tg128 |          6.33 ± 0.08 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | CPU        |       4 |           pp512 |         27.46 ± 2.31 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | CPU        |       4 |           tg128 |          9.50 ± 0.12 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | CPU        |       8 |           pp512 |         42.06 ± 2.22 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | CPU        |       8 |           tg128 |          9.92 ± 0.21 |

build: 458681e (1)
INFER_BENCH_DONE
