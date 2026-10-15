#!/usr/bin/env python3
"""Minimal vLLM CPU smoke test: does the source-built CPU backend actually run Qwen3.5-4B?"""
import os

os.environ["VLLM_TARGET_DEVICE"] = "cpu"
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "8"  # GiB for KV cache
os.environ.setdefault("HF_HOME", "/var/tmp/vllm-study/hf-cache")
# vLLM CPU requires tcmalloc preloaded (allocator perf). Set at the shell via LD_PRELOAD.

from vllm import LLM, SamplingParams


def main():
    llm = LLM(model="Qwen/Qwen3.5-4B", dtype="bfloat16",
              max_model_len=2048, enforce_eager=True,
              block_size=512)
    out = llm.generate(["The capital of France is"],
                       SamplingParams(max_tokens=16, temperature=0))
    print("GENERATED:", repr(out[0].outputs[0].text))
    print("CPU_SMOKE_OK")


if __name__ == "__main__":
    main()
