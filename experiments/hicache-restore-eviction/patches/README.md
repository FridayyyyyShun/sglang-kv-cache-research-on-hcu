# HCU MLA HiCache H2D JIT workaround

Based on section 5.2 of the HiCache T7/T8 report. Changes only two conditions in load_to_device_per_layer(), in sglang-das/python/sglang/srt/mem_cache/pool_host/mla.py:

```python
# Before
if self.can_use_jit:
# After
if self.can_use_jit and not _is_hcu:
```

HCU kernel H2D restores use the existing non-JIT layer_first/page_first implementations. Non-HCU and D2H conditions remain unchanged. This bypasses the reported Invalid address access / KERNEL VMFault path; it is not a proven kernel root-cause fix. No existing environment variable for this exact scope has been established.

## Apply

Stop your test server first. From the repository root containing sglang-das/:

```bash
git apply --check experiments/hicache-restore-eviction/patches/hcu-mla-h2d-disable-jit.patch
git apply experiments/hicache-restore-eviction/patches/hcu-mla-h2d-disable-jit.patch
git diff -- sglang-das/python/sglang/srt/mem_cache/pool_host/mla.py
```

Apply only after the check succeeds. A failed check may indicate a different source version or an already applied patch.

Alternatively, from the SGLang root containing python/:

```bash
git apply -p2 --check /absolute/path/hcu-mla-h2d-disable-jit.patch
git apply -p2 /absolute/path/hcu-mla-h2d-disable-jit.patch
```

Restart with --src repo and ensure M1_REPO points to the patched source python directory. M1_HICACHE_VERIFY controls fingerprint verification, not JIT. Run the T2 restore/fingerprint experiment to validate on your hardware.

## Revert

From the repository root containing sglang-das/:

```bash
git apply -R --check experiments/hicache-restore-eviction/patches/hcu-mla-h2d-disable-jit.patch
git apply -R experiments/hicache-restore-eviction/patches/hcu-mla-h2d-disable-jit.patch
```

Adjust the experiment folder name as above. From the SGLang root, add -p2. Restart after reverting.

## Validation

Passed git apply --check against the local colleague-repository clone. No GPU runtime test was performed here. Source files have not been modified. Committing the patch file alone does not modify the colleague source; applying it explicitly does.
