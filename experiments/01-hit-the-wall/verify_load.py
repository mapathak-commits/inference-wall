"""
Ampere gate check for Qwen3.5-4B. Before burning multi-hour sweeps, confirm the
hybrid model actually LOADS and RUNS on the A10G (sm_86): the Gated-DeltaNet /
linear-attention layers use Triton/Mamba-style kernels that have historically been
gated to newer GPUs (this series already found fp8 and two attention backends that
won't init on Ampere). Also print the memory split so we know the KV headroom.

Loads the engine offline (no server), generates one short completion, and reports:
  - that it initialized at all (no sm_86 kernel error),
  - weights vs KV-cache memory split (from vLLM's own logs / cache config),
  - a sanity token/s on a single prompt.
Usage: python verify_load.py <model_repo>
"""
import sys, time
from vllm import LLM, SamplingParams

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"

def main():
    t0 = time.time()
    # text-only serving of a multimodal model: keep max-model-len modest to match
    # the 0.5B serving study's workload envelope (256-in/128-out fits in 2048).
    llm = LLM(model=MODEL, dtype="float16", max_model_len=2048,
              max_num_seqs=256, gpu_memory_utilization=0.9,
              enforce_eager=False, trust_remote_code=True)
    load_s = time.time() - t0
    print(f"LOADED ok in {load_s:.1f}s", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)
    t1 = time.time()
    out = llm.generate(["Tell me about the number seven."], sp)
    gen_s = time.time() - t1
    ntok = len(out[0].outputs[0].token_ids)
    print(f"GEN ok: {ntok} tokens in {gen_s:.2f}s = {ntok/gen_s:.1f} tok/s (single stream)", flush=True)
    print("=== VERIFY DONE ===", flush=True)

# spawn start method (required so the forked EngineCore doesn't re-init CUDA)
# needs the entrypoint guarded, or children re-run this module on import.
if __name__ == "__main__":
    main()
