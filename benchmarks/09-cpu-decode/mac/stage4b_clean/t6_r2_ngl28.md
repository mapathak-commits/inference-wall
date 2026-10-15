| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |       6 |           pp512 |        411.73 ± 0.00 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |       6 |           tg128 |         52.94 ± 0.00 |

build: dfc29b6 (1)
