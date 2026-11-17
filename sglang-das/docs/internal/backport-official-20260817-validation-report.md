# Backport: official sglang main → HCU branch (20260817)

**Merge commit** `9f4368f10` · branch `backport/official-main-20260817`
**Range** `410088c91e` → `92b1d382c7` (`[Fix] Correct dense FP8 Marlin bias ordering (#35020)`, 2026-08-17)
**Size** 364 commits · 1397 files · +112 747 / −34 850 · 26 conflicts

The recorded backport base `410088c91e` is the natural merge base, so no `-s ours` anchoring was needed.

---

## 1. What came in

### Model support

| Area | Highlights |
|---|---|
| **Qwen3.5 / Qwen3.8** | GatedDeltaNet QKVZBA split/reshape/cat fused into one Triton kernel on HIP (#34421); grouped-head shared-KV verify acceleration (#34517); DeepEP-class backends + early EPLB state (#34810); MTP-with-HiCache startup fix (#34560); H20 fp8_w8a8 tuned configs (#34795) |
| **MiniMax M2.7 / M3 / H3** | Overlap shared and routed experts (#34542); NVFP4 routed TRT-LLM on SM100 (#32229); CPU optimization (#31956); w8a8 NPU adaptation (#32941); SubBlock training-free block-sparse DiT attention (#34148) |
| **GLM-5.2** | MXFP4 wide-EP16 2P1D nightly recipes (#34476) + cookbook (#34379); MoE weights restricted to local PP layers (#33793) |
| **New models** | Muse Glimmer (#34262); Gemma4 on Xeon (#22498); SANA-Video T2V (#32921); Krea-2 online FP8 (#34136) |
| **Diffusion** | 54 commits — the single largest area; bit-exact fused LayerNorm+modulate for FLUX.1 / GLM-Image / Sana / Z-Image, Wan causal VAE decode, LTX-2 RMSNorm+modulate, Cosmos qk-norm/RoPE/KV packing |

### Key technology

- **Speculative / DSpark** — logprobs support (#34696); MegaMoE for DSpark under DP attention (#34844); DSpark + DSV4 prefill-CP compatibility (#33865); DSpark shared-expert loading fix (#33312); unified-memory DSpark + two NaN root causes (page hand-out zeroing, CuTe); Inkling DSpark (#31847); three new tuning knobs `SGLANG_DSPARK_{FOLDED_PROPOSAL,STACKED_CTX_KV,EMBED_IN_GRAPH}`
- **DFLASH** — mamba-radix `extra_buffer_lazy` support (#34763); draft KV pool budgeted from its own attention geometry (#34234); sliding-attention causality defaults (#34524); DCP-aware draft KV sizing (#33912)
- **DCP (decode context parallel)** — one shared pack kernel across both a2a backends (#34651); fused a2a pack/unpack in MLA LSE reduce (#34614); two fewer per-layer launches in MLA target-verify (#34240)
- **PD disaggregation** — PP prefill with Mooncake staging buffer (#33807); `--enable-unified-memory` support (#33362); NIXL prefill bootstrap timeout (#34692); zmq socket cap via `SGLANG_DISAGGREGATION_ZMQ_MAX_SOCKETS`; skip speculative verify scratch on prefill servers
- **MoE / EP** — fused swiglu MoE up-GEMM epilogue (#32944); single-launch `moe_align` for tiny batches with many experts (#32395); explicit EPLB balancedness reporting (#34998); DeepEP v2 release workflow; W4AFP8 DeepEP scaling + mode-specific dtype fix (#33669)
- **KV cache / memory** — unified SWA page mapping in attention metadata (#35000); HiCache L2 transfer flattening (#34793); HiSparse shared-index plan-then-IO prefetch (#34329); SWA eviction frontier fix for bigram keys (#34870)
- **Kernel infrastructure** — JIT kernels moved fully into `namespace sglang`; per-token FP8 quantization migrated AOT → JIT (#34257); FlashInfer CuTe-DSL NVFP4 MoE quantization (#28354); architecture-owned SM12x FA4 kernels (#32991)

### Performance

`[DSpark] EP1 decode regression fix (#34759)` · `WAR read-done event published at DSPARK verify (#34816)` · `DSV4 host-sync removal in EAGLE prefill (#33662)` · `trivial DSV4 nonpaged indexer logits skipped (#33857)` · `DP-attention scheduler sync collapsed to one D2H copy (#34338)` · `paged_mqa_metadata optimized (#25855)` · `DSA indexer fp8-quant Q occupancy tuning (#32755)` · `FlashInfer MLA blocking D2H removed in spec-decode plan (#27689)` · `AITER unified-attention decode with scaled FP8 Q (#31856)`

---

## 2. Conflict resolution — 26 files

### Official direction adopted (local code superseded)

| File | Decision |
|---|---|
| `models/deepseek_v4.py` | Forward tail now **byte-identical to official**. The CP all-gather also re-gathers DSpark aux hidden states on the same token split (#33865), and `pre_hc_head` is unconditional. The local `need_pre_hc_head` gating is dropped — it saved only a tensor view and was the source of the earlier nextn shape mismatch. |
| `utils/cuda_ipc_transport_utils.py`, `multimodal/processors/base_processor.py` | Transport moved to `multimodal/transport/cuda_ipc.py`; the old file becomes official's compat shim. Official's `MmItemMemoryPool(pool_size, interval, base_gpu_id, tp_size)` + `wrap_tensor()` **supersedes** the local `MmItemMemoryPoolGroup` / `return_slices_with_flags`. |
| `disaggregation/decode_kvcache_offload_manager.py` | Official's `build_kv_host_pool()` refactor — verified it still routes MHA pools through `get_mha_host_pool_cls(kv_pool, hicache_mem_layout)`, so the HCU `layout_hcu` host pool is preserved. |
| `speculative/draft_utils.py` | Official's `("dsv4", backend)` tuple contract. |
| `layers/quantization/unquant.py` | `fuse_swiglu_interleaved` on the generic Triton fallback, HCU marlin/AITER W16A16 branches kept. |

### HCU behavior preserved

- **`eagle_worker_v2.py`** — keeps `need_hidden_states_before_norm` (official still hardcodes `return_hidden_states_before_norm=False`), combined with official's `get_schedule().page_size`.
- **`arg_groups/speculative_hook.py`** — DSpark device gate now allows CUDA / NPU / **HCU**.
- **`environ.py`** — rebuilt from official's file plus a dedicated HCU section carrying all **30** HCU-only knobs. Resolved by symbol diff rather than text merge, which also removed a pre-existing duplicate `SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL`.
- **`layers/moe/topk.py`** — keeps `num_token_non_padded` / `expert_location_dispatch_info` alongside official's `fused_shared_experts_scaling_factor`.
- `flashattention_backend`, `gdn_backend`, `quark_w4a4_mxfp4`, `attention_registry`, `fp8_kernel`, `server_args` — `_is_hcu` paths re-expressed on the new contracts.

### ⚠️ Default flips inherited from official (worth a reviewer's eye)

| Knob | was | now |
|---|---|---|
| `SGLANG_AUTO_NUMA_BIND` | `False` | `True` |
| `SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS` | `False` | `True` |
| `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` | `1024` | `8192` |

Neither was locally modified before (base == ours), so official's new default was taken. The MegaMoE one raises a buffer bound 8×; MegaMoE is not the a2a backend in either validated configuration.

Three overrides are deliberately kept and marked `# HCU override` inline: `SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0`, `SGLANG_HACK_FLASHMLA_BACKEND="kernel"`, `SGLANG_OPT_FP8_WO_A_GEMM=False`.

---

## 3. Build fix

Official's opt-in HIP batch copy in `kernels/aot/csrc/kvcacheio/transfer.cu` calls `hipMemcpyBatchAsync` under a plain runtime `if (kEnableHipBatch)`. Because that is a runtime — not `if constexpr` — branch, the body is still compiled even though the flag is `false`, and the symbol only exists from **HIP 7.0**. DTK 26.04 ships **HIP 6.3**, so the build failed with `use of undeclared identifier 'hipMemcpyBatchAsync'`.

The block is now gated on `HIP_VERSION >= 70000000` and falls back to the per-page copy, leaving official's code intact for ROCm 7.x.

---

## 4. Semantic audit

Two merge artifacts, present in neither parent, were found by ruff `F821/F401/F811` diffed against **both** the official target and pre-merge main (with line numbers normalized out — otherwise shifted pre-existing findings masquerade as new):

1. `base_processor.py` — still called `MmItemMemoryPoolGroup` with official's argument list (`F821`). Renamed to `MmItemMemoryPool`.
2. `draft_utils.py` — unused `ServerArgs` import (`F401`). Removed.

After the fixes: **zero findings absent from both parents.**

---

## 5. Validation

**Environment** `zz-sglang2` / `rye_sglang_0810`, eight HCUs idle at preflight, after a full `sgl-kernel` + `sglang` rebuild (`install_sglang.sh`, hipcc through gfx938).

### Static gates

| Gate | Result |
|---|---|
| Unmerged entries / conflict markers | none |
| `git diff --cached --check` | clean |
| Changed Python compiles | 1072 / 1072 |
| Key module imports (DSV4, nextn, dspark, eagle, environ, moe, mm) | 16 / 16 |
| `verify_hcu_registration.py` | OK |
| `check_hcu_runtime_text.py` | OK |
| `check_hcu_external_api_compat.py` | OK |
| ruff `F821,F401,F811` vs both parents | zero new |

### Runtime — DeepSeek-V4-Flash-0731-FP8-Channel

| Config | GSM8K (100q) | Invalid | Throughput | Faults |
|---|---|---|---|---|
| **Pure TP8** | **0.970** | 0.000 | 340.5 tok/s | none |
| **DSpark** | **0.960** | 0.000 | 378.7 tok/s | none |

Both produce coherent greedy output (`"The capital of France is"` → `" Paris. The capital of Spain is Madrid…"`); the DSpark run additionally serves a `temperature=0.8 / top_p=0.9` request without error, exercising the speculative renorm verify path. No VMFault, illegal access, or scheduler exception in either log. All eight cards returned to 0 % after shutdown.

**Reference:** pre-backport main measured 0.960 @ 306.4 tok/s on the same pure-TP configuration, so this backport is +1 pt accuracy and **+11 % throughput** on pure TP, with DSpark a further **+11 %** on top.

---

## 6. Status

Committed on `backport/official-main-20260817` as `9f4368f10` (parents `d95af36e1` + `92b1d382c7`). **Not pushed** — the two-parent shape keeps `92b1d382c7` discoverable as the base for the next backport.
