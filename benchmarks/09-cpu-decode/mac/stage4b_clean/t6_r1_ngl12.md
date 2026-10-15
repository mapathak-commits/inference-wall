| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |       6 |           pp512 |        197.55 ± 0.00 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |       6 |           tg128 |         39.65 ± 0.00 |

build: dfc29b6 (1)
