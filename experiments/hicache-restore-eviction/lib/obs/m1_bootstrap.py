"""被 .pth 行调用的引导模块。放在 obs 目录里，.pth 会把该目录加进 sys.path 后 import 本模块。

职责：装上 radix 观测钩子（import hook）。对主进程和 spawn 出来的 scheduler 子进程都生效，
因为 .pth 在每个 Python 解释器启动早期都会执行。
真正的监听逻辑在 sitecustomize.py，这里只负责"确保它被执行一次"。
"""
import os

# 只有在 handoff 文件存在（= 本轮用 serve.sh --obs 1 起过）时才挂钩子，避免污染无关进程
_HANDOFF = os.environ.get("M1_TRACE_HANDOFF", "/tmp/m1-radix/.trace_dir")
if os.path.exists(_HANDOFF) or os.environ.get("M1_RADIX_OBSERVE") == "1":
    try:
        import sitecustomize  # noqa: F401  （sitecustomize 自带 import hook，import 即安装）
    except Exception as _e:
        import sys
        sys.stderr.write("M1_BOOTSTRAP_ERROR %r\n" % (_e,))
