| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35 4B Q8_0                 |   4.29 GiB |     4.33 B | MTL,BLAS   |       1 |           pp512 |         37.66 ± 0.00 |
| qwen35 4B Q8_0                 |   4.29 GiB |     4.33 B | MTL,BLAS   |       1 |           tg128 |         15.40 ± 0.24 |
| qwen35 4B Q8_0                 |   4.29 GiB |     4.33 B | MTL,BLAS   |       2 |           pp512 |         73.37 ± 0.26 |
| qwen35 4B Q8_0                 |   4.29 GiB |     4.33 B | MTL,BLAS   |       2 |           tg128 |         26.82 ± 0.49 |
| qwen35 4B Q8_0                 |   4.29 GiB |     4.33 B | MTL,BLAS   |       4 |           pp512 |        142.35 ± 0.52 |
| qwen35 4B Q8_0                 |   4.29 GiB |     4.33 B | MTL,BLAS   |       4 |           tg128 |         40.58 ± 0.11 |
| qwen35 4B Q8_0                 |   4.29 GiB |     4.33 B | MTL,BLAS   |      10 |           pp512 |        280.55 ± 3.02 |
| qwen35 4B Q8_0                 |   4.29 GiB |     4.33 B | MTL,BLAS   |      10 |           tg128 |         43.15 ± 0.33 |

build: dfc29b6 (1)
