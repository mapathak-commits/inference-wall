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

New parts are published weekly, on Fridays.

**Live:**

| Part | Title | Status |
|---|---|---|
| Primer | [How an LLM actually serves a request](articles/primer.md) | published 2026-08-22 |
| Primer 2 | [What actually happens inside an LLM](articles/primer-2.md) | published 2026-09-08 — reference companion |
| 1 | [An 8.6 GB model that serves only 7 requests a second](articles/part-1.md) | published 2026-08-22 |
| 2 | [The prefill that freezes your decoders](articles/part-2.md) | published 2026-08-28 |
| 3 | [The batching cliff](articles/part-3.md) | published 2026-09-04 |

**Drafted, not yet live** (branch → recommended publish slot):

| Slot | Title | Branch | State |
|---|---|---|---|
| Field note | I opened up one prompt to see what the model was thinking | `preview/fn2` | reviewed, ready |
| 4 | Starving the cache: how a server degrades when it runs out of KV | `preview/part-4-corrected` | full draft (article form) |
| 5 | Quantization as a fit-enabler | `preview/part-5` | full draft |
| 6 | Speculative decoding | `preview/part-6` | full draft |
| 7 | FlashAttention at the scale where it matters | — | **not yet drafted** |
| 8 | Off the rig: decode on a CPU | untracked `articles/part-8.md` | full draft |

### Recommended publishing order

The order is content-driven, not just chronological, and the drafts' own
cross-references pin most of it:

1. **Field note (attention internals)** — ready now, standalone, depends only on
   Primer 2 (live). Ship it next: it forward-references "Part 5, still to come"
   and "FlashAttention, a coming post," so publishing it before those keeps both
   references honest, and it is a light detour between the batching cliff and the
   heavier cache/quantization posts. Unnumbered, like the primers, so it never
   blocks the numbered arc.
2. **Part 4 — Starving the cache.** The natural continuation of the arc: its
   finding (a starved cache degrades into a small-batch server) is priced by Part
   3's cliff, and it completes the two-walls picture — Part 1's bandwidth wall
   plus this capacity wall.
3. **Part 5 — Quantization.** Closes the original arc that 1→3→4 build (fewer
   bytes per token attacks both walls at once). The draft calls itself the finale
   of that arc.
4. **Part 6 — Speculative decoding.** Must follow 4 and 5: the draft explicitly
   leans on Part 4's admission-control signature and references Part 5 and Part
   3's crossover.
5. **Part 7 — FlashAttention** *(needs drafting)*. Generalizes the
   bytes-through-memory law one level down and is the first part to run a dense
   model alongside the rig. This is the one gap: there is no draft yet, so it is
   the sequencing risk if the CPU post is ready first.
6. **Part 8 — Off the rig (CPU).** Complete and depends only on Part 1, so it can
   slot anywhere after Part 1, but reads best as the "after the arc closes"
   excursion. Candidates beyond it: prefix caching, tooling, cross-engine
   comparisons.

**Two things to resolve before shipping in this order:**

- **The 7-before-8 gap.** Part 8 (CPU) is drafted but Part 7 (FlashAttention) is
  not. Either draft FlashAttention before shipping the CPU post, or renumber the
  CPU post to 7 (which then requires editing Part 4's body, below).
- **Part 4 ↔ Part 8 reference.** Part 4's body cites "the same lesson Part 8
  arrived at from the other direction" in the past tense, which reads as if Part 8
  already shipped. Publishing 4 before 8 needs that reworded to neutral/present
  tense (or publish 8 first).

Also worth a cleanup: `preview/part-6` still carries stale `drafts/part-2/` and
`drafts/part-3/` copies of already-published posts; and Part 4's draft is in
article form (Liquid paths) on its branch rather than the `drafts/<part>/README.md`
preview convention the other drafts use.

Primer 2 ("What actually happens inside an LLM") is a reference companion,
not a numbered part: it opens the attention black box the first primer left shut
(query/key/value, the KV cache falling out of them, stacking blocks, and
next-token prediction), has no benchmarks, and stops before any finding. It
ships alongside Part 3; it will be linked from the first primer's footer and
from the parts that lean on the attention picture it draws, rather than inserted
into the weekly arc as a numbered part.

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

## Publishing a new part (weekly)

The release checklist for each part, so every week is the same mechanical step:

1. **Article.** Adapt the reviewed draft into `articles/part-N.md`:
   - front matter with `title` and `permalink: /articles/part-N/`; a byline line
     (*Part N of "The Inference Wall". Same rig...*) instead of the draft header,
     followed by an author/date line (*Manas Pathak · Month D, YYYY*)
   - replace every `<!-- FIGURE ... -->` / `<!-- DIAGRAM ... -->` marker with an image
     embed from `assets/` (descriptive alt text, `relative_url` filter)
   - rewrite the reproduce footer to point at this repo's
     `experiments/0N-.../` and `benchmarks/0N-.../` folders
   - prev/next navigation links and the disclaimer at the bottom
2. **Code and data.** Copy the part's scripts into `experiments/0N-<slug>/` and its raw
   logs/traces into `benchmarks/0N-<slug>/`, each with a README mapping file → claim.
   Shared server-launch scripts go in `scripts/`.
3. **Assets.** Add the part's figures to `assets/figures/` (descriptive names) and
   diagrams to `assets/diagrams/` (recompress to ~1400 px JPEG, quality 80).
4. **Index.** In `index.md` and this README's series table: flip the part's status from
   *coming* to a link. Update the previous part's "Next" navigation to link the new one.
5. Commit and push; GitHub Pages redeploys automatically.

### Previewing a draft before it ships

Drafts are reviewed on `preview/<part>` branches, never on `main` (Pages builds only
from `main`, so nothing on a preview branch reaches the live site — but the repo is
public, so a draft branch is technically visible to anyone who goes looking). The
pattern: put the draft and its images in `drafts/<part>/` on the branch, name the
markdown `README.md` so GitHub renders it on the folder view, keep image paths plain
and relative so they resolve in GitHub's renderer. Review at
`github.com/<owner>/<repo>/blob/preview/<part>/drafts/<part>/README.md`, then delete
the branch once the part is published. Liquid tags (`relative_url`) don't render on
GitHub blob views, so drafts use plain paths and are converted at publish time.

Publication-ready parts wait on `publish/<part>` branches; going live is a
fast-forward merge of that branch into `main`. Unlike previews, a publish branch
carries the real article with Liquid paths plus its `experiments/`, `benchmarks/`,
and `assets/` material, and the index/nav flips.

Current preview branches: `preview/fn2` (attention-internals field note, ready),
`preview/part-4-corrected` (Starving the cache — note this one holds an
article-form draft, not the `drafts/<part>/README.md` convention),
`preview/part-5` (quantization), `preview/part-6` (speculative decoding). See the
recommended publishing order under [The series](#the-series).

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
