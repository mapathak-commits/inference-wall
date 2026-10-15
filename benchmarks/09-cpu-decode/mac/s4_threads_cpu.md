| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |       1 |           pp512 |         28.01 ± 0.04 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |       1 |           tg128 |         11.88 ± 0.04 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |       2 |           pp512 |         54.94 ± 0.05 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |       2 |           tg128 |         21.90 ± 0.11 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |       3 |           pp512 |         79.67 ± 0.13 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |       3 |           tg128 |         30.72 ± 0.16 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |       4 |           pp512 |        104.75 ± 0.27 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |       4 |           tg128 |         37.99 ± 0.41 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |       5 |           pp512 |        122.75 ± 1.23 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |       5 |           tg128 |         42.74 ± 0.34 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |       6 |           pp512 |        145.34 ± 1.07 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |       6 |           tg128 |         45.85 ± 0.33 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |       7 |           pp512 |        165.38 ± 1.48 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |       7 |           tg128 |         48.62 ± 1.45 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |       8 |           pp512 |        187.70 ± 1.86 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |       8 |           tg128 |         50.71 ± 3.02 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |       9 |           pp512 |        202.39 ± 0.33 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |       9 |           tg128 |         51.61 ± 3.45 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |      10 |           pp512 |        209.73 ± 2.57 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |      10 |           tg128 |         47.54 ± 2.79 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |      12 |           pp512 |        187.85 ± 3.32 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |      12 |           tg128 |         23.30 ± 1.76 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |      14 |           pp512 |        173.97 ± 2.55 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |      14 |           tg128 |         13.80 ± 1.60 |

build: dfc29b6 (1)
