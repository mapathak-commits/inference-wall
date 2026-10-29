---
title: "Off the rig, part two: what a faster memory bus reveals"
permalink: /articles/part-10/
image: /assets/figures/fig9e-gap-vs-size.png
---

*Part 10 of "The Inference Wall", the second half of the CPU study: Qwen3.5-4B and two
smaller models on a cloud server slice and a MacBook with an M4 Pro chip, each with its
own GPU
alongside. Draft.*

*Manas Pathak · draft, October 2026*

[The Inference Wall]({{ '/' | relative_url }}) · [All posts]({{ '/articles/' | relative_url }}) · **Part 10**

[Part 9]({{ '/articles/part-9/' | relative_url }}) carried the series' claim onto two
CPUs and found it held: for one user, the time to produce a token is the model's bytes
divided by the memory bandwidth, plus a fixed cost per token, and on a 4B model the first
term is nearly the whole token on both machines. This part is about the second term, the
one Part 9 set aside.

```
time per token  =  weight bytes / memory bandwidth  +  fixed cost per token
```

On the server slice, streaming the weights takes a quarter of a second and anything that
does not scale with bytes is invisible beside it. On the MacBook the same stream
takes seven times less, and the things it used to hide come into view: a fixed cost that is a tenth or more of a small model's token, a per-format compute cost that the earlier fit
mistook for that fixed cost, a gap between CPU and GPU that bends as models get small,
and two practical rules for unified-memory machines that run the opposite way from
discrete-GPU habits. Along the way it backs up an answer Part 9 gave in a single sentence, to
the question of whether the MacBook is fast because of its CPU or because of its memory.

## What the fast machine makes visible

Part 9 fitted that line to the MacBook's four weight formats, the weights stored at 4, 6,
8, or 16 bits each, called Q4, Q6, Q8, and F16, and read the bandwidth off the slope. The same fit produced intercepts, 2.4 ms for the GPU and about 4.6 ms for the
CPU, which looked like a difference in fixed cost but turned out not to be one. The follow-up
with smaller models below, which holds the weight format fixed, puts the fixed cost near 1 ms
on both engines. What the ladder fit was most likely absorbing into its intercept is a third thing the
one-line equation leaves out: a per-format compute cost.
Four-bit and eight-bit weights must be unpacked before use, and that costs more
instructions per byte than F16, so the quantized points sit a little above the line that
the F16 points define. On the ladder, format changed together with size, so the format
cost looked like a constant.

That is also why the simple one-line claim starts to leak on fast memory. On the server
slice, streaming 3 GB of Q4 weights takes about 91 ms, so a few milliseconds of fixed and
format cost are a rounding error and effective bandwidth comes out constant across weight
sizes: 34, 32, and 30 GB/s for bf16, Q8, and Q4, all at the 33 GB/s wall. On the MacBook,
streaming the same 3 GB takes about 14 ms, the fixed and format costs did not shrink with
it, and effective bandwidth is no longer constant, rising from 157 GB/s at Q4 to 188 GB/s
at F16 on the CPU within the same session. The faster the memory, the larger the share of
a token everything other than streaming takes, and the sooner it becomes the thing worth
optimizing. The constant-effective-bandwidth check is a slow-machine diagnostic; once
memory is fast, the line with an intercept is the right model, and the intercept has to
be measured with the format held fixed.

## Does model size change the gap?

In the equation, model size cancels out of the ratio between two chips as long as the
first term dominates, so the gap should not depend on size until the fixed cost does.
Two smaller models, Qwen2.5 at 0.5B and 1.5B, in five files across several formats, were
run on both machines with the same engine and a same-session 4B anchor. These smaller models keep separate
tables for the words they read in and the words they write out, and the input table is
looked up one row per token rather than streamed, so their files overstate the bytes read
per token by 14 to 29%. The 4B shares one table for both, so its file size is accurate. Fitting the line on streamed bytes with
the format held fixed gives each chip's two terms:

| | Memory bandwidth, from the fit | Fixed cost per token |
|---|---|---|
| Server slice CPU | 33 GB/s | 3.1 ms |
| A10G | 503 GB/s | 1.1 ms |
| MacBook CPU | 209 to 222 GB/s | 0.9 to 1.2 ms |
| MacBook GPU | 208 to 232 GB/s | 0.8 to 1.2 ms |

The fixed cost is about a millisecond on three of the four chips and about three
milliseconds on the server CPU; why the slice's is larger was not measured. The gap at each size follows:

| Model | Streamed GB | Server GPU / CPU | MacBook GPU / CPU, two sessions |
|---|---|---|---|
| 0.5B Q4 | 0.39 | 7.4x | 0.90x, 0.95x |
| 0.5B Q8 | 0.53 | 8.4x | 0.86x, 0.90x |
| 1.5B Q4 | 0.98 | 10.0x | 1.07x, 1.06x |
| 0.5B F16 | 0.99 | 10.8x | 1.03x, 0.99x |
| 1.5B F16 | 3.09 | 13.7x | 1.10x, 1.11x |
| 4B F16 | 8.66 | 14.3x | 0.99x, 1.04x |

![Decode gap against streamed model bytes on both machines: the server's GPU-over-CPU ratio falls from 14x at 8.7 GB toward 7x at 0.4 GB, while the MacBook's ratio stays near 1x and dips below it for the smallest quantized files](../assets/figures/fig9e-gap-vs-size.png)

On the server the gap halves as the model shrinks, from 14.3x to 7.4x, even though the
CPU's fixed cost is the larger one in milliseconds. The equation says why: what matters is
each chip's fixed cost as a share of its own streaming time. At 0.4 GB the A10G streams
the weights in under a millisecond and then spends a millisecond on fixed work, so its
token is more than half overhead, while the CPU's 3 ms is a fifth of its 12 ms stream. As the model shrinks further the gap tends toward the ratio of the two fixed costs, under
3x, and the GPU's bandwidth advantage stops counting. On the MacBook the two fixed costs are equal, so the gap stays near 1x at every size for F16, and the two engines tie at the
4B anchor, 0.99x and 1.04x across sessions. The small quantized files are the one
departure: there the CPU wins, 0.86 to 0.95x, because the GPU's four-bit and eight-bit kernels, the programs that run one operation
each, fall further below their own line than the CPU's do on a model this small. The likely
reason is that the work is too small to fill the GPU, but that was not measured.

So the answer is that model size changes the gap only through the second term, and on
both machines the second term favors the CPU at small sizes. Part 9's 13.5x and 1.09x are the large-model numbers, where the first term is nearly
the whole token.

## Is the MacBook fast, or is its memory?

Part 9 found the MacBook's CPU decoding the F16 4B about 5.4x faster than the server
slice's CPU, and two ratios could explain that. Their memory bandwidths differ by 7.2x, 238
against 33 GB/s. Their matrix math rates differ by about 4.8x, 3.1 against 0.65 trillion
operations a second. On that number alone you cannot tell which one is doing the work.

Three measurements settle which one it is, starting with the fit: on the MacBook the CPU
and the GPU decode on the same line
with the same slope while differing 2.7x in math rate; if math set the speed the lines
would differ. The MacBook's decode stops speeding up after 4 or 5 of its 10 performance cores while
prefill keeps scaling to 9 or 10; adding cores adds math and does nothing for decode. And the bf16 file
shows what happens when math is the limit: the MacBook CPU is stuck at 4 tok/s, the same
as the server, until a kernel that can keep up with the bus is swapped in. So the MacBook's lead does not come from its faster cores. It comes from Apple putting a
memory bus as wide as a GPU's under a CPU. The shortfall from 7.2x to 5.4x is not the fixed
cost, which at about a millisecond is only a few percent of a 4B token. It is that the
slice decodes at its memory ceiling, 35 GB/s effective against 33 measured, while the
MacBook's CPU reaches about four fifths of its own, 188 against 238.

The server slice is not a slow CPU either, but a narrow slice of one socket's memory
system. A full EPYC socket with all eight memory channels in use delivers roughly 200
GB/s and would land near the MacBook; a newer twelve-channel server socket at roughly 460 GB/s
would beat it and approach the A10G. Neither was measured here, but
the point stands as arithmetic: the thing to ask about a CPU is how wide its memory bus is, and "server" or
"laptop" does not answer it.

## What unified memory changes: energy, and a rule that flips

On the MacBook the GPU is 1.09x faster at decode than the CPU across the 4B ladder, and
at F16 the two tie. That invites the conclusion that the GPU is pointless for decode on
a unified-memory machine. It is wrong at the 4B size, for a reason no throughput table
shows, and the reason weakens as the model shrinks.

Measured with Apple's power-monitoring tool while decoding, the CPU run draws about 42 W above
idle and the GPU run about 26 W, of which 21.5 W is the GPU itself and the rest the CPU
feeding it. For nearly the same tokens per second, the GPU does the work for **about
1.75x less energy per token**, 1.78x and 1.73x in two independent captures. The CPU run
pins all ten performance cores at 100% against the chip's power ceiling; the same work on
the GPU leaves the CPU at a fraction of that power. At 0.5B Q4 the advantage shrinks to 1.18x, 112 against 132 millijoules per token, because at about 260 tokens a second the CPU spends 13.6 W just encoding and submitting
the GPU's commands. So
the engine choice on this machine has no bandwidth-based answer and a power-based one
that depends on size: for a 4B model, decode belongs on the GPU for battery and heat; for
a tiny one, it barely matters where it runs.

One more unified-memory result, because it flips a discrete-GPU habit. On a normal GPU
box you offload as many layers to the GPU as its memory holds. On the MacBook, moving layers from the CPU to the GPU a few at a time made decode slower across most of the
range: any split from 4 to 16 of the 32 layers ran at 0.80 to 0.87x of all-CPU,
break-even came at about 20 layers, and from 24 layers up
it was faster, reaching 1.18x at full offload.
No bytes cross a bus here, so the cost must be somewhere in the handoff between two engines sharing one memory; it
measures about 7 ms per token and is not yet explained. The rule on unified
memory is to pick one engine and commit; partial offload exists to cope with limited GPU
memory, and this machine has no such limit.

## What to take away

1. **The fixed cost per token is about a millisecond on the A10G and on both MacBook
   engines, and three on the server CPU.** It has to be measured with the weight format
   held fixed; a ladder that changes format with size reads a per-format compute cost as a
   constant, which is how an earlier fit got 4.6 ms.
2. **Model size moves the gap only through the fixed cost, and toward the CPU.** On the
   server the gap halves from 14x to 7x between 8.7 GB and 0.4 GB because the GPU's
   millisecond is more than half of its tiny token. On the MacBook the gap stays near 1x at
   every size for F16 and the CPU wins on the smallest quantized files.
3. **The MacBook's lead over the cloud CPU comes from its memory bus rather than its cores.**
   Same slope as its own GPU, decode saturating at half the cores, and a kernel-bound case
   that erases the lead entirely. By the channel arithmetic, a full server socket would
   close most of the gap, though that was not measured.
4. **On unified memory, throughput does not decide the engine; power does, and less so
   for small models.** At 4B the GPU is 1.09x faster and 1.75x more energy-efficient at
   decode; at 0.5B the energy edge is 1.18x.
5. **On unified memory, splitting the model across engines never beat putting all of it
   on the better one.** Splits from 4 to 16 layers run at 0.80 to 0.87x of all-CPU, and
   even the deeper splits that beat the CPU stay below full GPU offload; pick one engine
   and commit.

---

*Reproduce: the small-model follow-up ran the official `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
and `Qwen/Qwen2.5-1.5B-Instruct-GGUF` files, byte-identical on both machines, with a
same-session 4B anchor, through `llama-bench` on the same builds as Part 9, 458681e on
the server slice and dfc29b6 on the M4 Pro. Streamed bytes exclude the untied embedding
table, read from the GGUF headers. The M4 Pro runs were repeated on an idle machine and
both sessions are reported. Power is from `powermetrics --samplers cpu_power,gpu_power`,
two captures at 4B and one at 0.5B Q4. The offload sweep is three round-robin rounds at
`-t 6`. The server scripts and the figure renderer are in
[`experiments/09-cpu-decode/`](https://github.com/mapathak-commits/inference-wall/tree/main/experiments/09-cpu-decode)
with the Mac protocol described in that folder's README; the raw tables, the small-model
analysis, the 4B power summaries, the Mac results file that carries the 0.5B energy
capture, and the Mac study's audit ledger are in
[`benchmarks/09-cpu-decode/`](https://github.com/mapathak-commits/inference-wall/tree/main/benchmarks/09-cpu-decode),
shared with Part 9.*

---

**Previous:** [Part 9: Off the rig, decode on a CPU, where the memory bus decides]({{ '/articles/part-9/' | relative_url }}) · [All posts]({{ '/articles/' | relative_url }})

---

*Disclaimer: This blog is written and published in my personal capacity. The opinions,
findings, and conclusions expressed herein are solely my own and do not necessarily
represent the views, policies, or endorsements of my current or past employers.*
