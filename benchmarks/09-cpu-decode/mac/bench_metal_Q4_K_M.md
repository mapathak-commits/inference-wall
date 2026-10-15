| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |      10 |           pp512 |        769.02 ± 0.35 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | MTL,BLAS   |      10 |           tg128 |         57.49 ± 0.05 |

build: dfc29b6 (1)
