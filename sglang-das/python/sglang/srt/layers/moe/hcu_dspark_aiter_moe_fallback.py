"""Temporary HCU fallback for the unsupported DSpark draft MoE geometry.

Delete this module and its two call-site hooks after AITER provides the
gfx938 FP8 channel-wise config for E=257, inter_dim=256, hidden=4096, topk=7.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
from typing import Any, Optional

from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

_FALLBACK_ENV = "SGLANG_HCU_DSPARK_AITER_MOE_FALLBACK"
_fallback_enabled = get_bool_env_var(_FALLBACK_ENV, default="true")
_force_triton = contextvars.ContextVar(
    "hcu_dspark_aiter_moe_force_triton", default=False
)
_warned_geometries: set[tuple[int, ...]] = set()


def is_triton_forced_for_dspark_aiter_fallback() -> bool:
    """Whether the current synchronous MoE invocation must bypass AITER."""
    return _force_triton.get()


@contextlib.contextmanager
def _force_triton_for_one_call():
    token = _force_triton.set(True)
    try:
        yield
    finally:
        _force_triton.reset(token)


def try_run_dspark_aiter_moe_triton_fallback(
    *,
    runner: Any,
    dispatch_output: Any,
    layer: Any,
    M: int,
    N1: int,
    N2: int,
    K: int,
    E: int,
    top_k: int,
    use_shuffle: bool,
) -> Optional[Any]:
    """Run Triton only for the known unsupported DSpark draft MoE geometry.

    Returning ``None`` asks the caller to preserve the original AITER error.
    This keeps unrelated missing configs and unsafe shuffled layouts visible.
    """
    if not _fallback_enabled:
        return None

    geometry = (N1, N2, K, E, top_k)
    if geometry != (512, 4096, 4096, 257, 7):
        return None
    if use_shuffle or not runner.runner_backend.is_triton():
        return None

    if geometry not in _warned_geometries:
        logger.warning(
            "AITER FP8 MoE has no DSpark draft backend for first observed "
            "M=%d, N1=%d, N2=%d, K=%d, E=%d, topk=%d; using the temporary "
            "Triton fallback. Set %s=0 to require full AITER coverage.",
            M,
            *geometry,
            _FALLBACK_ENV,
        )
        _warned_geometries.add(geometry)

    # Keep the workaround self-contained. Importing lazily also avoids a cycle
    # with triton_utils.fused_moe, which consults the ContextVar above.
    from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo

    quant_info = TritonMoeQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        use_fp8_w8a8=True,
        per_channel_quant=True,
        w13_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        a13_scale=layer.w13_input_scale,
        a2_scale=layer.w2_input_scale,
    )
    with _force_triton_for_one_call():
        return runner.run(dispatch_output, quant_info)
