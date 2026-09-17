# The Inference Wall

**Understanding ML inference by breaking it apart.**

A blog series, published with GitHub Pages at
**<https://mapathak-commits.github.io/inference-wall/>**, plus the code and raw data behind
every number in it.

The method, in one line: take one real model on one ordinary GPU, turn a single knob until
something breaks, report the before-and-after number, and read the profiler trace that
explains why. A break, a number, and a trace.

The rig: `Qwen/Qwen3.5-4B` (fp16, with a `Qwen3.5-9B-AWQ` companion for the quantization
post) on a single NVIDIA A10G (23 GB), served by vLLM 0.18.0 and measured under real load
with `vllm bench serve`.

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
| 4 | [The attention sink: why deep layers pour their weight onto the first token](articles/part-4.md) | published 2026-09-18 |

**Drafted, not yet live** (publish order = part number; branches carry the draft):

| Part | Title | Branch | State |
|---|---|---|---|
| 5 | Starving the cache: how a server degrades when it runs out of KV | `preview/part-4-corrected` | full draft (article form) |
| 6 | Quantization as a fit-enabler | `preview/part-5` | full draft |
| 7 | Speculative decoding | `preview/part-6` | full draft |
| 8 | FlashAttention at the scale where it matters | — | **not yet drafted** |
| 9 | Off the rig: decode on a CPU | `preview/part-9` | full draft |

> **Branch names lag the numbering.** The renumbering (below) shifted every draft
> down by one, but the `preview/<part>` branch names still read as they were first
> cut: `preview/part-4-corrected` → Part 5, `preview/part-5` → Part 6, `preview/part-6`
> → Part 7, `preview/part-9` → Part 9. Each draft's own body carries the correct new
> number; the branch name is just a stable handle. (Part 4, the attention-internals
> post, shipped from `publish/part-4` and is now live.)

### The renumbering (2026-09-13)

The attention-internals post began life as an unnumbered "field note," a detour
outside the weekly arc. It was folded into the numbered series as **Part 4** to
avoid introducing a third content category beyond *post* and *primer*. That shifted
every later draft down by one:

| Post | Was | Now |
|---|---|---|
| Attention internals (was "field note") | — | **4** |
| Starving the cache | 4 | **5** |
| Quantization | 5 | **6** |
| Speculative decoding | 6 | **7** |
| FlashAttention *(undrafted)* | 7 | **8** |
| Off the rig (CPU) | 8 | **9** |

Every draft's self-reference and cross-references were updated on its branch to
match. The "finale" framing the quantization post once carried was also dropped:
the series has grown past a fixed five-post arc, so no single post closes it.

### Recommended publishing order

The order is content-driven, not just chronological, and equals the part numbers
above. The drafts' own cross-references pin most of it:

1. **Part 4 — Attention internals** *(published 2026-09-18)*. Standalone, depended
   only on Primer 2. It forward-references "Part 6, still to come" (quantization)
   and "FlashAttention, a coming post," so shipping it before those keeps both
   references honest, and it is a light detour between the batching cliff and the
   heavier cache/quantization posts.
2. **Part 5 — Starving the cache.** Its finding (a starved cache degrades into a
   small-batch server) is priced by Part 3's cliff, and it completes the two-walls
   picture — Part 1's bandwidth wall plus this capacity wall.
3. **Part 6 — Quantization.** Fewer bytes per token attacks both walls at once;
   ties back to where the series started.
4. **Part 7 — Speculative decoding.** Must follow 5 and 6: the draft explicitly
   leans on the cache post's admission-control signature (Part 5) and references
   quantization (Part 6) and Part 3's crossover.
5. **Part 8 — FlashAttention** *(needs drafting)*. Generalizes the
   bytes-through-memory law one level down and is the first part to run a dense
   model alongside the rig. This is the one gap: there is no draft yet, so it is
   the sequencing risk if the CPU post is ready first.
6. **Part 9 — Off the rig (CPU).** Complete and depends only on Part 1, so it can
   slot anywhere after Part 1, but reads best as the "after the arc closes"
   excursion. Candidates beyond it: prefix caching, tooling, cross-engine
   comparisons.

**One thing to resolve before shipping in this order:** Part 8 (FlashAttention)
is not yet drafted while Part 9 (CPU) is complete. Either draft FlashAttention
before shipping the CPU post, or accept the CPU post shipping ahead of its number.

Also worth noting: Part 5's draft is in article form (Liquid paths) on its branch
rather than the `drafts/<part>/README.md` preview convention the other drafts use.

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

Current preview branches (branch name → part number; see the renumbering note under
[The series](#the-series)): `preview/part-4-corrected` (Part 5, Starving the cache —
note this one holds an article-form draft, not the `drafts/<part>/README.md`
convention), `preview/part-5` (Part 6, quantization), `preview/part-6` (Part 7,
speculative decoding), `preview/part-9` (Part 9, decode on a CPU). See the recommended
publishing order under [The series](#the-series). (`preview/fn2` shipped as Part 4 and
can be deleted.)

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
