# Experiments 07 — Speculative decoding

Scripts behind [Part 7](../../articles/part-7.md). Everything ran on a single NVIDIA A10G
(23 GB) serving Qwen3.5-4B in fp16 through vLLM 0.18.0. The speculator is vLLM's `ngram`
method (prompt lookup decoding), run at depth k5 and k3 against a spec-off baseline;
nothing else changes between arms.

| File | What it does |
|---|---|
| `run_spec_study.sh` | main A/B driver: brings up a server per arm (`off`, `ngram-k5`, `ngram-k3`), runs a discarded warm pass, the single-stream and 32-concurrent probes, then the standard 256/128 rate sweep. |
| `run_spec_followup.sh` | follow-up driver: the k5/off stability re-runs at flood and the predictable-decoder trace capture. |
| `spec_probes.py` | the single-stream (temperature 0, 160-token) and 32-concurrent probes for predictable vs creative prompts; prints `PROBE` lines with ms/token and aggregate tok/s. |
| `capture_spec_trace.py` | drives `/start_profile` and `/stop_profile` around eight long-lived predictable decoders to capture a bounded profiler window with no arrivals. |
| `tp_spec_kernels.py` | kernel-family census of the verify-step trace: counts and CUDA time per kernel, plus the decode-vs-verify path split for the linear-attention layers. Args: `TRACE`. |
| `tp_spec_steps.py` | reconstructs engine-step cadence from the delta-rule kernel bursts and reports per-step time and GPU-busy fraction. Args: `TRACE`. |
| `render_spec_figures.py` | renders `fig7a-sweep-inversion.png` and `fig7b-seat-collapse.png` from the sweep table and the startup/scheduler logs. |
| `render_spec_animation.py` | renders `fig7-token-stream.gif` from the measured per-prompt arrival rates. |
| `render_spec_cartoon.py` | vector fallback for the d9 speculation diagram. |

## What "ngram" is

The speculator keeps no model and no n-gram statistics. It is a backward string search
over the request's own prompt-plus-output (`prompt_lookup_min`..`prompt_lookup_max` phrase
lengths); on a hit it proposes the k tokens that followed the phrase last time. S's cost is
pinned at zero, so everything measured is the cost and benefit of L's verification
machinery.

## Reproduce

Run `run_spec_study.sh` for the sweep and probes, `run_spec_followup.sh` for the stability
re-runs and the trace. The trace analyzers take the gzipped trace in
`benchmarks/07-speculative-decoding/traces/`; gunzip first or point them at the `.json`.
