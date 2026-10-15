# CPU optimization sweep — Q4_K_M — 2026-09-01T00:43:25Z

### 4 cores, 1 CCX (0-3), strict
| model                          |       size |     params | backend    | threads | cpu_mask   | cpu_strict |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------- | ---------: | --------------: | -------------------: |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | CPU        |       4 | 0x0F       |          1 |           pp512 |         29.92 ± 1.22 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | CPU        |       4 | 0x0F       |          1 |           tg128 |          9.80 ± 0.09 |

build: 458681e (1)

### 4 cores, split CCX (0,1,4,5)
| model                          |       size |     params | backend    | threads | cpu_mask   | cpu_strict |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------- | ---------: | --------------: | -------------------: |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | CPU        |       4 | 0x33       |          1 |           pp512 |         26.44 ± 0.93 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | CPU        |       4 | 0x33       |          1 |           tg128 |          9.45 ± 0.16 |

build: 458681e (1)

### 8 physical cores (0-7), strict
| model                          |       size |     params | backend    | threads | cpu_mask   | cpu_strict |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------- | ---------: | --------------: | -------------------: |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | CPU        |       8 | 0xFF       |          1 |           pp512 |         47.48 ± 1.16 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | CPU        |       8 | 0xFF       |          1 |           tg128 |          9.75 ± 0.20 |

build: 458681e (1)

### 8 threads, default placement
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | CPU        |       8 |           pp512 |         40.31 ± 2.81 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | CPU        |       8 |           tg128 |          9.96 ± 0.23 |

build: 458681e (1)

### 8 phys, KV q8_0
| model                          |       size |     params | backend    | threads | cpu_mask   | cpu_strict | type_k | type_v |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------- | ---------: | -----: | -----: | --------------: | -------------------: |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | CPU        |       8 | 0xFF       |          1 |   q8_0 |   q8_0 |           pp512 |         45.30 ± 0.73 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | CPU        |       8 | 0xFF       |          1 |   q8_0 |   q8_0 |           tg128 |          9.70 ± 0.21 |

build: 458681e (1)

### 8 phys, flash-attn on
| model                          |       size |     params | backend    | threads | cpu_mask   | cpu_strict |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------- | ---------: | --: | --------------: | -------------------: |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | CPU        |       8 | 0xFF       |          1 |   1 |           pp512 |         45.58 ± 1.69 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | CPU        |       8 | 0xFF       |          1 |   1 |           tg128 |          9.78 ± 0.22 |

build: 458681e (1)

### 8 phys, poll 100
| model                          |       size |     params | backend    | threads | cpu_mask   | cpu_strict |       poll |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------- | ---------: | ---------: | --------------: | -------------------: |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | CPU        |       8 | 0xFF       |          1 |        100 |           pp512 |         46.99 ± 1.78 |
| qwen35 4B Q4_K - Medium        |   2.80 GiB |     4.33 B | CPU        |       8 | 0xFF       |          1 |        100 |           tg128 |          9.71 ± 0.17 |

build: 458681e (1)

OPT_SWEEP_DONE
