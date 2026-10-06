#!/usr/bin/env python3
"""
Render the Part 6 token-stream animation as an animated GIF (GitHub-Pages-safe:
plain markdown image embed, no JS).

Two terminal panes, both against the ngram-k5 server, driven by the MEASURED rates:
  top    predictable text  11.2 ms/token, arriving in verify-step bursts
                            (36.2 ms steps, mean ~3.2 accepted tokens/step)
  bottom creative text      19.4 ms/token, one token at a time (guesses rejected)
Playback is slowed 4x (a real 0.8 s of generation is unwatchable); the clocks shown
are REAL milliseconds. Token text is the models' actual temperature-0 output heads
(PROBETEXT lines in spec_study.log).
"""
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
MONO = ImageFont.truetype(FONT_DIR + "DejaVuSansMono.ttf", 14)
MONO_B = ImageFont.truetype(FONT_DIR + "DejaVuSansMono-Bold.ttf", 14)
SMALL = ImageFont.truetype(FONT_DIR + "DejaVuSansMono.ttf", 12)

W, PANE_H, HDR, FTR = 880, 168, 34, 26
H = HDR + 2 * PANE_H + FTR
BG, PANE_BG = (17, 20, 24), (24, 28, 34)
DIM, NEW, TXT, HL = (110, 190, 130), (255, 200, 80), (225, 228, 232), (140, 160, 180)

# Real temperature-0 output heads from spec_study.log (PROBETEXT), whitespace-split.
PRED = (" The quick brown fox jumps over the lazy dog. The quick brown sentence. "
        "<think> Thinking Process: 1. **Analyze the Request:** * Task: Repeat a "
        "specific sentence exactly 20 times.0 times. * Sentence: 'The quick brown "
        "fox jumps over the lazy dog.' * Constraint check: exact repetition, no "
        "numbering. 2. **Drafting:** The quick brown fox jumps over the lazy "
        "dog.").split()
CREA = ("<think> Here's a thinking process that leads to the poem: 1. **Analyze "
        "the Request:** * **Topic:** A clockwork octopus dreaming in colors. * "
        "**Genre: Surreal poem. * **Key Elements:** brass, gears, ticking, ink, "
        "sleep, spectra. 2. **Brainstorming imagery:** verdigris tentacles, "
        "escapement hearts, chromatic dreams, wound springs.").split()

NTOK = 40                      # tokens animated per pane
MS_STEP_PRED = 36.2            # verify-step period (trace, p50)
BURSTS = [4, 3, 3, 4, 2, 4, 3, 3, 4, 2, 4, 3]   # mean 3.25 -> ~11.2 ms/token
MS_TOK_CREA = 19.4             # measured single-stream creative ms/token
DILATE = 4                     # playback slowdown
FRAME_MS = 20                  # real ms per frame -> GIF delay = 80 ms

def arrivals_pred():
    t, i, out = 0.0, 0, []
    b = 0
    while i < NTOK:
        t += MS_STEP_PRED
        n = BURSTS[b % len(BURSTS)]; b += 1
        for _ in range(min(n, NTOK - i)):
            out.append(t); i += 1
    return out

def arrivals_crea():
    return [(i + 1) * MS_TOK_CREA for i in range(NTOK)]

ARR = {"pred": arrivals_pred(), "crea": arrivals_crea()}
DONE = {k: v[-1] for k, v in ARR.items()}
TOTAL_REAL = max(DONE.values()) + 400          # trailing hold
FRAMES = int(TOTAL_REAL / FRAME_MS) + 1

def wrap(words, width=100):
    lines, cur = [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if len(cand) > width:
            lines.append(cur); cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines

def draw_pane(d, y0, title, words, arr, now, mstok_label):
    d.rectangle([8, y0, W - 8, y0 + PANE_H - 6], fill=PANE_BG)
    d.text((18, y0 + 6), title, font=MONO_B, fill=HL)
    shown = sum(1 for a in arr if a <= now)
    fresh = sum(1 for a in arr if now - FRAME_MS < a <= now)
    old, new = words[:shown - fresh], words[shown - fresh:shown]
    y = y0 + 30
    lines = wrap(old + new)
    n_new = len(new)
    flat = " ".join(old + new)
    # draw wrapped, coloring the trailing n_new words amber
    total_words = len(old) + len(new)
    wi = 0
    for line in lines[-6:]:
        x = 18
        for w in line.split():
            wi += 1
        y += 0
    # simpler: draw line by line, tracking word index
    wi = 0
    start_line = max(0, len(lines) - 6)
    wi = sum(len(l.split()) for l in lines[:start_line])
    for line in lines[start_line:]:
        x = 18
        for w in line.split():
            wi += 1
            color = NEW if wi > total_words - n_new else DIM
            d.text((x, y), w, font=MONO, fill=color)
            x += MONO.getlength(w + " ")
        y += 19
    t_shown = min(now, arr[-1])
    status = f"{shown:2d}/{NTOK} tokens   {t_shown:5.0f} ms"
    if now >= arr[-1]:
        status += f"   DONE — {mstok_label}"
    d.text((18, y0 + PANE_H - 28), status, font=MONO_B,
           fill=NEW if now < arr[-1] + 1 else TXT)

frames = []
for f in range(FRAMES):
    now = f * FRAME_MS
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((12, 8), "ngram-k5 speculation, one request each — token arrival, real "
                    "clocks, playback slowed 4x", font=MONO_B, fill=TXT)
    draw_pane(d, HDR, "predictable text — guesses accepted in bursts",
              PRED, ARR["pred"], now, "11.2 ms/token (1.8x)")
    draw_pane(d, HDR + PANE_H, "creative text — guesses rejected, token-by-token",
              CREA, ARR["crea"], now, "19.4 ms/token (no gain)")
    d.text((12, H - 20), "measured on Qwen3.5-4B / A10G / vLLM 0.18.0 — "
                         "amber = tokens that arrived this instant",
           font=SMALL, fill=HL)
    frames.append(im)

frames[0].save("spec-token-stream.gif", save_all=True, append_images=frames[1:],
               duration=FRAME_MS * DILATE, loop=0, optimize=True)
print(f"wrote spec-token-stream.gif: {FRAMES} frames, "
      f"{TOTAL_REAL:.0f} ms real, {TOTAL_REAL*DILATE/1000:.1f} s playback")
