# Benchmarks 08 — FlashAttention at the scale where it matters

Raw measurement logs and profiler traces behind [Part 8](../../articles/part-8.md).
Nothing here is post-processed; the numbers in the post are transcribed from these files.
Single A10G, Qwen2.5-7B-Instruct, fp16, single stream.

| File | Produced by | Backs |
|---|---|---|
| `s1.log` | `s1_probe.py` (both arms) | the prefill TTFT table (256..30720) and the decode ITL table. Quad coefficients: FLASH 3,689 vs FLEX 10,099 ps/n². Decode ITL +11% (FLASH) vs +159% (FLEX). |
| `s0.log` | `s0_prefill_sweep.py` | S0 de-risk: superlinear prefill bend confirmed on the A10G. |
| `s3.log` | `run_s3.sh` | S3 capture driver log: which window was profiled at each length. |
| `server_s1_7b_flash.log`, `server_s1_7b_flex.log` | `start_server.sh` | the two server configs, each logging its selected backend ("Using Flash Attention backend" / "Using FlexAttention backend on V1 engine"). |
| `traces/{flash,flex}_prefill_{2k,8k,28k}.json.gz` | `s3_capture.py` | prefill profiler windows. Matmul time is byte-identical between backends; attention grows to 34% (FLASH) vs 58% (FLEX) of the window at 28k. |
| `traces/{flash,flex}_decode_{2k,28k}.json.gz` | `s3_capture.py` | decode profiler windows. FLASH runs `flash_fwd_splitkv_kernel`; FLEX runs a single monolithic `triton_tem_fused_0` with no split-KV. |

## The kernel-family census

Run `experiments/08-flash-attention/tp_census.py` against any trace for its attention /
matmul / other split. The 28k prefill windows are the headline: matmul is ~5,755 ms either
backend, while attention is 3,098 ms (FLASH) against 8,389 ms (FLEX), a 2.7x kernel gap that
equals the entire wall-clock gap between the two windows.

## Reading the traces yourself

No GPU needed. Gunzip and open in `chrome://tracing` or
[ui.perfetto.dev](https://ui.perfetto.dev), or run `tp_census.py` from
`experiments/08-flash-attention/` against the `.json.gz` for the kernel counts quoted in
the post. Each GPU kernel is a chrome-trace event with a category, a timestamp, and a
duration.
