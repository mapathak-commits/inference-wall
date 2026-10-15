# Small-model decode: EPYC slice CPU vs A10G, llama.cpp 458681e, 2026-10-15T01:20:12Z

## File sizes (bytes)
- models/small/qwen2.5-0.5b-instruct-fp16.gguf: 1266425696
- models/small/qwen2.5-0.5b-instruct-q4_k_m.gguf: 491400032
- models/small/qwen2.5-0.5b-instruct-q8_0.gguf: 675710816
- models/small/qwen2.5-1.5b-instruct-fp16.gguf: 3560416288
- models/small/qwen2.5-1.5b-instruct-q4_k_m.gguf: 1117320736

## qwen2.5-0.5b-instruct-fp16.gguf
### CPU, -ngl 0, 8 physical cores pinned
| model                          |       size |     params | backend    | threads | cpu_mask   | cpu_strict |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------- | ---------: | --------------: | -------------------: |
| qwen2 1B F16                   |   1.17 GiB |   630.17 M | CPU        |       8 | 0xFF       |          1 |           pp128 |       389.70 ± 54.39 |
| qwen2 1B F16                   |   1.17 GiB |   630.17 M | CPU        |       8 | 0xFF       |          1 |           tg128 |         31.42 ± 0.75 |

build: 458681e (1)
### A10G, -ngl 99
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen2 1B F16                   |   1.17 GiB |   630.17 M | CUDA       |  99 |           pp128 |   16273.99 ± 8069.07 |
| qwen2 1B F16                   |   1.17 GiB |   630.17 M | CUDA       |  99 |           tg128 |        339.59 ± 1.40 |

build: 458681e (1)

## qwen2.5-0.5b-instruct-q4_k_m.gguf
### CPU, -ngl 0, 8 physical cores pinned
| model                          |       size |     params | backend    | threads | cpu_mask   | cpu_strict |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------- | ---------: | --------------: | -------------------: |
| qwen2 1B Q4_K - Medium         | 462.96 MiB |   630.17 M | CPU        |       8 | 0xFF       |          1 |           pp128 |       318.96 ± 10.74 |
| qwen2 1B Q4_K - Medium         | 462.96 MiB |   630.17 M | CPU        |       8 | 0xFF       |          1 |           tg128 |         69.33 ± 1.41 |

build: 458681e (1)
### A10G, -ngl 99
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen2 1B Q4_K - Medium         | 462.96 MiB |   630.17 M | CUDA       |  99 |           pp128 |   11648.73 ± 3237.17 |
| qwen2 1B Q4_K - Medium         | 462.96 MiB |   630.17 M | CUDA       |  99 |           tg128 |        511.28 ± 4.53 |

build: 458681e (1)

## qwen2.5-0.5b-instruct-q8_0.gguf
### CPU, -ngl 0, 8 physical cores pinned
| model                          |       size |     params | backend    | threads | cpu_mask   | cpu_strict |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------- | ---------: | --------------: | -------------------: |
| qwen2 1B Q8_0                  | 638.74 MiB |   630.17 M | CPU        |       8 | 0xFF       |          1 |           pp128 |       372.30 ± 40.62 |
| qwen2 1B Q8_0                  | 638.74 MiB |   630.17 M | CPU        |       8 | 0xFF       |          1 |           tg128 |         56.17 ± 2.04 |

build: 458681e (1)
### A10G, -ngl 99
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen2 1B Q8_0                  | 638.74 MiB |   630.17 M | CUDA       |  99 |           pp128 |   12434.00 ± 3471.71 |
| qwen2 1B Q8_0                  | 638.74 MiB |   630.17 M | CUDA       |  99 |           tg128 |        471.46 ± 1.79 |

build: 458681e (1)

## qwen2.5-1.5b-instruct-fp16.gguf
### CPU, -ngl 0, 8 physical cores pinned
| model                          |       size |     params | backend    | threads | cpu_mask   | cpu_strict |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------- | ---------: | --------------: | -------------------: |
| qwen2 1.5B F16                 |   3.31 GiB |     1.78 B | CPU        |       8 | 0xFF       |          1 |           pp128 |        123.71 ± 5.72 |
| qwen2 1.5B F16                 |   3.31 GiB |     1.78 B | CPU        |       8 | 0xFF       |          1 |           tg128 |         10.23 ± 0.57 |

build: 458681e (1)
### A10G, -ngl 99
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen2 1.5B F16                 |   3.31 GiB |     1.78 B | CUDA       |  99 |           pp128 |    7490.86 ± 1908.86 |
| qwen2 1.5B F16                 |   3.31 GiB |     1.78 B | CUDA       |  99 |           tg128 |        139.73 ± 0.43 |

build: 458681e (1)

## qwen2.5-1.5b-instruct-q4_k_m.gguf
### CPU, -ngl 0, 8 physical cores pinned
| model                          |       size |     params | backend    | threads | cpu_mask   | cpu_strict |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------- | ---------: | --------------: | -------------------: |
| qwen2 1.5B Q4_K - Medium       |   1.04 GiB |     1.78 B | CPU        |       8 | 0xFF       |          1 |           pp128 |        160.64 ± 2.89 |
| qwen2 1.5B Q4_K - Medium       |   1.04 GiB |     1.78 B | CPU        |       8 | 0xFF       |          1 |           tg128 |         29.47 ± 0.83 |

build: 458681e (1)
### A10G, -ngl 99
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen2 1.5B Q4_K - Medium       |   1.04 GiB |     1.78 B | CUDA       |  99 |           pp128 |    7204.70 ± 1471.87 |
| qwen2 1.5B Q4_K - Medium       |   1.04 GiB |     1.78 B | CUDA       |  99 |           tg128 |        293.76 ± 1.10 |

build: 458681e (1)

## 4B F16 anchor, same session
### CPU
| model                          |       size |     params | backend    | threads | cpu_mask   | cpu_strict |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ---------- | ---------: | --------------: | -------------------: |
| qwen35 4B F16                  |   8.06 GiB |     4.33 B | CPU        |       8 | 0xFF       |          1 |           pp128 |         43.05 ± 0.27 |
| qwen35 4B F16                  |   8.06 GiB |     4.33 B | CPU        |       8 | 0xFF       |          1 |           tg128 |          3.80 ± 0.06 |

build: 458681e (1)
### A10G
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 4B F16                  |   8.06 GiB |     4.33 B | CUDA       |  99 |           pp128 |     3180.15 ± 507.38 |
| qwen35 4B F16                  |   8.06 GiB |     4.33 B | CUDA       |  99 |           tg128 |         54.34 ± 0.03 |

build: 458681e (1)
SMALL_DONE 2026-10-15T01:24:42Z
