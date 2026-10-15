| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35 4B BF16                 |   8.06 GiB |     4.33 B | MTL,BLAS   |       1 |           pp512 |        120.14 ± 2.75 |
| qwen35 4B BF16                 |   8.06 GiB |     4.33 B | MTL,BLAS   |       1 |           tg128 |          0.46 ± 0.01 |
| qwen35 4B BF16                 |   8.06 GiB |     4.33 B | MTL,BLAS   |       2 |           pp512 |        190.82 ± 1.88 |
| qwen35 4B BF16                 |   8.06 GiB |     4.33 B | MTL,BLAS   |       2 |           tg128 |          0.96 ± 0.00 |
| qwen35 4B BF16                 |   8.06 GiB |     4.33 B | MTL,BLAS   |       4 |           pp512 |        244.62 ± 1.88 |
| qwen35 4B BF16                 |   8.06 GiB |     4.33 B | MTL,BLAS   |       4 |           tg128 |          1.72 ± 0.00 |
| qwen35 4B BF16                 |   8.06 GiB |     4.33 B | MTL,BLAS   |      10 |           pp512 |        211.65 ± 2.07 |
| qwen35 4B BF16                 |   8.06 GiB |     4.33 B | MTL,BLAS   |      10 |           tg128 |          4.09 ± 0.01 |

build: dfc29b6 (1)
