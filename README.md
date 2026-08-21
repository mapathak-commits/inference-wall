# The Inference Wall

**Understanding ML inference by breaking it apart.**

A blog series, published with GitHub Pages at
**<https://mapathak-commits.github.io/inference-wall/>**, plus the code and raw data behind
every number in it.

The method, in one line: take one real model on one ordinary GPU, turn a single knob until
something breaks, report the before-and-after number, and read the profiler trace that
explains why. A break, a number, and a trace.

The rig: `Qwen/Qwen3.5-4B` (fp16, with a `Qwen3.5-9B-AWQ` companion for the finale) on a
single NVIDIA A10G (23 GB), served by vLLM 0.18.0 and measured under real load with
`vllm bench serve`.

## The series

New parts are published weekly.

| Part | Title | Status |
|---|---|---|
| Primer | [How an LLM actually serves a request](articles/primer.md) | published |
| 1 | [An 8.6 GB model that serves only 7 requests a second](articles/part-1.md) | published |
| 2 | The prefill that freezes your decoders | coming |
| 3 | The batching cliff | coming |
| 4 | Starving the cache | coming |
| 5 | Quantization as a fit-enabler | coming |

## Layout

```
articles/      the blog posts (rendered by GitHub Pages)
experiments/   the benchmark/probe scripts behind each post, one folder per post
benchmarks/    the raw measurement logs and profiler traces each post cites
notebooks/     analysis notebooks (as they are published)
scripts/       shared server-launch scripts used across experiments
assets/        figures and diagrams embedded in the posts
```

Every number in a published post is backed by a raw log or trace in `benchmarks/` and was
produced by a script in `experiments/` or `scripts/`. Folders for a post land together with
the post.

## Reproducing

Python 3.12 and a CUDA-12.x GPU. The version pins in `requirements.txt` matter (Qwen3.5 is
not registered in vLLM before 0.18.0, and 0.18.0's default wheel pulls a CUDA-13 torch):

```bash
python -m venv venv && . venv/bin/activate
pip install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The scripts hardcode a model cache and venv path from the original box; grep them for
`/var/tmp` and point them at your own locations. Every launch needs
`VLLM_WORKER_MULTIPROC_METHOD=spawn` (already baked into the scripts).

The captured traces in `benchmarks/*/traces/` need no GPU at all: gunzip and open in
`chrome://tracing` or [ui.perfetto.dev](https://ui.perfetto.dev).

## Disclaimer

This blog is written and published in my personal capacity. The opinions, findings, and
conclusions expressed herein are solely my own and do not necessarily represent the views,
policies, or endorsements of my current or past employers.

## Author

Manas Pathak (<mapathak@gmail.com>)

## License

[MIT](LICENSE)
