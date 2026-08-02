---
name: picking-amd-gpu-backend
description: >-
  Chooses and installs the right GPU compute backend (vulkan, rocm, cpu) for
  local AI on an AMD GPU, then verifies the workload actually landed on the
  discrete GPU. Use when the user is setting up local image generation, local
  LLM inference, llama.cpp, stable-diffusion.cpp, or Lemonade on AMD hardware;
  when a ROCm or HIP install fails; when they mention TheRock, gfx1100/gfx1102/
  gfx1151, Radeon RX, eGPU, or an external GPU enclosure; when local AI is
  unexpectedly slow, falls back to CPU, or runs on the wrong GPU; or when they
  ask whether they need ROCm at all. Also use when generation fails above a
  certain resolution, or when a machine has both an integrated and a discrete
  GPU. Do not use for AMD Instinct data center parts (see
  serving-llms-on-instinct) or for NVIDIA hardware.
---

# Picking an AMD GPU backend

Consumer AMD GPUs have three backend options for local AI. Picking wrong costs
hours: ROCm currently fails to install on Windows, and several settings that
look like optimizations are large regressions.

**Default to Vulkan.** Reach for ROCm only on Linux, or after Vulkan is proven
working and you have a specific reason.

## Decision

```
Is this AMD Instinct (MI300X/MI325X/MI350X)?
  -> yes: stop, use the serving-llms-on-instinct skill instead.

Is the OS Windows?
  -> yes, and the target is native Windows: use vulkan. The Lemonade/TheRock
     ROCm backend install currently fails there (see below).
  -> yes, and the target is WSL2: rocm is a real option, not just a fallback
     — see the WSL2 note below. Still confirm the specific GPU model, since
     package-name guessing is the main way this goes wrong.
  -> no (bare-metal Linux): try rocm; fall back to vulkan if install or
     runtime fails. [UNVERIFIED ON BARE METAL -- see note]

No GPU detected, or GPU unsupported?
  -> use cpu, and tell the user the expected slowdown (~20x for images).
```

> **Bare-metal Linux is unverified; WSL2 is verified working — for LLM
> inference. Image generation is verified too, and is slower there.** Every
> measurement in the Windows branch is measured on real hardware. WSL2 ROCm
> was also measured directly — a real HIP kernel compiled and ran correctly,
> llama.cpp built with the HIP backend matched Vulkan's throughput on the
> same 8B model (36.4 vs 33-34 tok/s), but stable-diffusion.cpp built with
> the HIP backend (`SD_HIPBLAS`) generated a real, visually-checked image
> **~3-4x slower** than the Vulkan warm-server baseline (10.1s vs 2.47s,
> SD-Turbo 512²) — all on the same Radeon RX 7600M XT (gfx1102). Don't
> assume ROCm/WSL2 is a blanket win just because LLM throughput matched;
> check the specific workload. Full repro steps, and every wrong turn taken
> to get there (including two Windows-host-to-WSL2 scripting footguns that
> have nothing to do with the GPU), are at
> <https://github.com/gucciwong/amd-local-ai-bench/blob/main/docs/rocm-on-wsl2.md>.
>
> **Correction to an earlier version of this note: PyTorch training on WSL2
> ROCm does work, it just needs one non-obvious fix.** A ROCm nightly PyTorch
> install on this same WSL2 setup initially got `torch.cuda.is_available() ==
> False` (WSL2 exposes only `/dev/dxg`, no `/dev/kfd`/`/sys/class/kfd`), and
> an earlier version of this note called that a dead end. It wasn't: the pip
> wheel bundles its own Linux-only `libhsa-runtime64.so`, which shadows the
> system ROCm's copy — and the system copy (from `amdrocm-wsl`) is already
> linked against AMD's WSL bridge, `librocdxg.so`. Deleting the wheel's
> bundled `torch/lib/libhsa-runtime64.so*` (and the same file under
> `triton/backends/amd/lib/`) so the loader falls through to the system
> library flips `is_available()` to `True`. Verified with real training, not
> just device enumeration: a 4096-dim MLP and an fp16-autocast pass both
> converged correctly (loss decreasing, GPU-vs-CPU matmul results matching)
> on the same Radeon RX 7600M XT (gfx1102). If a user wants to train/fine-tune
> on this GPU on Windows (e.g. via Unsloth's new AMD support), WSL2 is a live
> option — point them at this fix first, not at giving up on WSL2.
>
> **Unsloth's AMD support is now verified working on this exact GPU at three
> model scales, not just theoretically compatible** — real QLoRA fine-tunes
> converged at each: Llama-3.2-1B (1.33 GB peak VRAM), Qwen3-4B (3.89 GB),
> and Qwen3-8B (7.83 GB — on an 8176 MiB card, ~345 MB of headroom left, no
> OOM). Those numbers land almost exactly on AMD's own published targets for
> this hardware tier (`amd/gaia#667`: "~4GB" and "~8GB, right at the ceiling"
> for the same two model sizes) — this is the first independent confirmation
> of those targets on real gfx1102 hardware. One extra footgun along the way:
> `pip install unsloth[amd]` silently replaces the already-fixed ROCm torch
> with a plain PyPI CUDA build (`torch-2.11.0+cu130`, pulling in `nvidia-*`
> packages on a machine with no NVIDIA GPU at all) — `torch.cuda.is_available()`
> goes back to `False` with no error. Fix: reinstall torch from the ROCm index
> at the version range Unsloth's own docs specify, then redo the
> `libhsa-runtime64.so` deletion (a fresh torch install brings back its own
> bundled copy every time). This was also checked against a false-positive
> risk: loss decreasing alone doesn't rule out "the base model already knew
> this" — so the real verification used facts the model can't have seen in
> pretraining (invented place names, currencies, a flag color). Before
> fine-tuning it correctly said it didn't know any of them; after 60 steps
> on 4 such facts it recalled all 4 verbatim. That's real evidence of
> learning, not a loss-curve artifact.
>
> By contrast, a from-scratch fair comparison against `torch-directml`
> (native Windows, no fix needed to install) using plain `transformers`+`peft`
> LoRA — chosen because DirectML has no Unsloth/bitsandbytes support —
> completed 3 clean, reproducible runs on WSL2/ROCm (283-350ms/step, 3823 MB
> peak VRAM) but **crashed on DirectML on a real Llama model**, every time,
> regardless of precision or attention implementation:
> `RuntimeError: value cannot be converted to type uint8_t without overflow`,
> inside `transformers`' causal-mask preparation (`masked_fill`). Tried fp16,
> fp32, and `attn_implementation="eager"` — identical error each time. This is
> a known, permanently-closed issue —
> [microsoft/DirectML#702](https://github.com/microsoft/DirectML/issues/702),
> closed `not_planned` ("DirectML is in maintenance mode"), independently
> corroborated on a different model and GPU in a
> [Gemma-3-1b-it HF discussion](https://huggingface.co/google/gemma-3-1b-it/discussions/19)
> — so it isn't specific to Llama or this card, and there's no fix coming.
> **The issue's own suggested workaround (replace `masked_fill` with an
> equivalent `torch.where` call) was tested directly** — monkey-patched
> `LlamaModel._prepare_4d_causal_attention_mask_with_cache_position` and
> reran the same training. The crash genuinely disappears, but training then
> produces `loss=nan` from step 0 onward — the workaround only removes the
> symptom that makes the program stop; there's a deeper problem underneath.
>
> **That deeper problem was traced to its actual root, and it's neither
> `masked_fill` nor `torch.where`.** Per-module forward hooks pinpointed the
> first NaN at layer 0's `self_attn.o_proj` output — meaning attention itself
> already produced NaN. `transformers` has a safety call,
> `AttentionMaskConverter._unmask_unattended`, specifically to prevent NaN
> from fully-masked attention rows, but it's gated by
> `attention_mask.device.type in ["cuda", "xpu"]` — DirectML's device type is
> `"privateuseone"`, so that safety net is silently skipped. Adding the call
> unconditionally threw a new error — `causal_mask` was already a
> `BoolTensor` by that point, not float. Tracing that back further: original
> (unpatched) `transformers` code does `causal_mask *= (bool comparison
> tensor)`, an in-place `float *= bool`. A minimal isolated repro confirms
> **DirectML's multiply kernel does not follow standard type promotion for
> `float * bool`** (true for both in-place and out-of-place) — it silently
> degrades the *entire result* to bool dtype, discarding the float values.
> That's the true, single root cause: it corrupts `causal_mask` before
> `masked_fill` ever runs, explaining both the original crash and the NaN
> that survived `torch.where`.
>
> **The real fix is simpler than the community workaround and doesn't touch
> `masked_fill` at all**: cast the bool comparison to the mask's dtype before
> multiplying (`keep_mask = (arange > cache_position).to(dtype)`, then
> `causal_mask = causal_mask * keep_mask`). With only that change — original
> `masked_fill` unchanged — training ran 15 clean steps, loss 4.04 → 1.26, no
> crash, no NaN. This is more precise than
> [microsoft/DirectML#702](https://github.com/microsoft/DirectML/issues/702)'s
> own suggested fix, which only addresses the downstream symptom. Don't tell
> a user "patch this and it's fixed" for the `torch.where` workaround alone —
> it silences the crash without producing a model that actually trains. Don't
> assume `torch-directml`'s easier install means it's the safer bet for real
> model training on this GPU today. Filed the type-promotion root cause
> upstream as [microsoft/DirectML#737](https://github.com/microsoft/DirectML/issues/737)
> and cross-linked it from #702 — neither has a maintainer response yet.
>
> Full writeup, including the wrong "dead end" conclusion this replaces and
> why it was wrong, is at
> <https://github.com/gucciwong/amd-local-ai-bench/blob/main/docs/training-methodologies.md>.
>
> Bare-metal Linux (no WSL) follows upstream documentation only — no one has
> run this skill's guidance on real bare-metal Linux. Say so when recommending
> it there, and prefer vulkan as the fallback the moment anything fails rather
> than debugging a ROCm install.
>
> **If a ROCm install on WSL2 looks like it's failing, don't conclude the GPU
> is unsupported before checking these two things** — both cost real time in
> the referenced session:
> - `amdgpu-install --usecase=wsl` errors with "not supported or invalid"
>   because that usecase name does not exist in current amdgpu-install
>   builds, not because WSL itself is unsupported. The actual bridge is a
>   separate package, `amdrocm-wsl`, installed alongside the versioned ROCm
>   package.
> - Per-architecture package names carry a ROCm version suffix
>   (`amdrocm7.14-gfx1102`, not `amdrocm-gfx1102`). Run
>   `apt-cache search amdrocm | grep <gfx-target>` before concluding a
>   package "doesn't exist" for a given GPU.

## Step 1: identify the hardware before installing anything

Never assume which GPU will be used. Machines with an integrated *and* a
discrete GPU are common, and the integrated one is often enumerated first.

```bash
lemonade backends            # which backends are installable for this hardware
```

A backend shown as `installable` means hardware filtering already accepts it.
`unsupported` messages name the reason (`Requires AMD XDNA 2 AMD NPU`,
`Unsupported GPU`, `Requires Linux`).

To see how the compute library enumerates devices — the order matters, because
index 0 is not necessarily the discrete GPU:

```bash
llama-server --list-devices
```

Look for `matrix cores`. On RDNA3/4 the discrete GPU reports `KHR_coopmat`;
integrated GPUs typically report `none`. Prefer the device with matrix cores.

## Step 2: install the backend

```bash
lemonade backends install <recipe>:vulkan
lemonade config set sdcpp.backend=vulkan
lemonade config set llamacpp.backend=vulkan
```

Vulkan installs in seconds because it ships with the AMD display driver.

### If the user insists on ROCm on Windows

Set expectations first: on Windows this currently downloads several GB and then
fails at extraction:

```
Error: TheRock extraction failed: bin directory not found
```

This is an upstream packaging bug, tracked in
[lemonade-sdk/lemonade#2722](https://github.com/lemonade-sdk/lemonade/issues/2722),
reproduced on more than one GPU architecture. `--force` does not help — it
bypasses hardware filtering, which is not the blocker. There is no CLI flag to
pin an alternate ROCm version.

Vulkan reaches the same matrix-core hardware, so the practical cost of skipping
ROCm on Windows is low.

## Step 3: verify the workload landed on the discrete GPU

A backend that "works" may still be running on the integrated GPU. Confirm by
watching dedicated VRAM rise when a model loads and fall when it unloads.

On Windows:

```powershell
Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory |
  Select-Object Name, DedicatedUsage, SharedUsage
```

The discrete GPU is the adapter with non-zero `DedicatedUsage`; integrated GPUs
are UMA and report 0 dedicated with large `SharedUsage`. Load a model, re-run,
and confirm the expected adapter grew.

If dedicated VRAM never moves, the work is on the wrong device or on CPU.

## Settings that look helpful and are not

State these before the user finds them:

| Setting | Reality |
|---|---|
| `enable_dgpu_gtt=true` | Lets the driver allocate from system memory. On a discrete GPU — especially over Thunderbolt — measured 58x slower image generation, and it does **not** raise the resolution ceiling. Keep `false`. |
| Splitting one job across two GPUs | `sd-cpp` accepts `--backend clip=...,vae=...,diffusion=...`. On a discrete + integrated pair this regressed at every resolution measured. Integrated GPUs lack matrix cores; cross-device transfer is pure overhead. |
| `max_loaded_models=2` | No benefit if the two models do not both fit in VRAM — they evict each other anyway. Shrink the models instead. |
| `GGML_VK_FORCE_MAX_ALLOCATION_SIZE` | Bounded near 4 GB by a uint32 limit, so it can only *lower* the cap. It cannot raise a 5 GiB per-allocation limit. |

## When generation fails above a certain resolution

If image generation fails abruptly past some size with:

```
ggml_vulkan: Device memory allocation of size 5368709120 failed
```

that is Vulkan's **per-allocation** cap (5 GiB here), not total VRAM. Adding
memory does not help — quantized models and system-memory borrowing were both
measured and neither moved the ceiling.

Recommend the upscale path instead of fighting it: generate small, then run a
pure upscale. This is dramatically faster than generating large directly, and
uses a fraction of the VRAM.

```bash
sd-cli -M upscale --upscale-model <esrgan.pth> -i small.png -o large.png
```

Note that `--hires` is *not* this: it runs a second full denoising pass, so it
costs about as much as generating at the target size.

## Before reporting any performance number

Two mistakes invalidate measurements on this hardware. Both are easy to repeat.

**Align arguments across execution paths.** A server may launch its worker with
flags the CLI does not default to (e.g. `--vae-tiling --diffusion-fa`). Comparing
the two without matching flags produces conclusions that point the wrong
direction. Capture what actually ran:

```powershell
Get-CimInstance Win32_Process -Filter "Name='sd-server.exe'" | Select-Object CommandLine
```

**Clear VRAM first.** Two processes each wanting 6-7 GB on an 8 GB card
regressed one workload by 34x. Unload resident models before benchmarking, and
restart the server between configurations.

```bash
lemonade unload <model>
```

## Reference

Measured numbers, the hardware they came from, and the negative results behind
each recommendation are in the `local-ai-use` skill's
[hardware-notes.md](../local-ai-use/hardware-notes.md).
