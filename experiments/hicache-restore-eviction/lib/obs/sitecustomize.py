"""M0/M1 共用的 import hook。只在 p_m1.sh / serve.sh 显式打开开关时生效。

M1 现场用的是 UnifiedRadixCache（registry 默认），不是旧的 RadixCache。
所以 M1 监视的模块是 sglang.srt.mem_cache.unified_radix_cache，
模块加载完后把观测钩子挂到 module.UnifiedRadixCache 上。
"""
import os


def _trace_dir():
    """M0_TRACE_DIR 优先；spawn 子进程丢了环境变量时，读 handoff 文件兜底。"""
    d = os.environ.get("M0_TRACE_DIR")
    if d:
        return d
    try:
        with open(os.environ.get("M1_TRACE_HANDOFF", "/tmp/m1-radix/.trace_dir")) as f:
            d = f.read().strip()
        return d if d and os.path.isdir(d) else None
    except Exception:
        return None


# M1 开关：环境变量有就用；没有但 handoff 文件在，也认为开启（兼容 spawn 子进程）
_HAS_TRACE = _trace_dir()
M0 = os.environ.get("M0_OBSERVE") == "1" and _HAS_TRACE
M1 = (os.environ.get("M1_RADIX_OBSERVE", "1") != "0") and _HAS_TRACE

if M0 or M1:
    import sys, importlib.abc, importlib.machinery

    WATCH_M0 = [
        "sglang.srt.model_executor.model_runner",
        "sglang.srt.layers.attention.dsa.dsa_indexer",
    ]
    # [M1] 现场默认用 UnifiedRadixCache；旧 RadixCache 也挂上做兜底（万一 registry 切回）
    WATCH_RADIX = [
        "sglang.srt.mem_cache.unified_radix_cache",
        "sglang.srt.mem_cache.radix_cache",
    ]

    WATCHED = set()
    if M0:
        WATCHED.update(WATCH_M0)
    if M1:
        WATCHED.update(WATCH_RADIX)

    class M0Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in WATCHED:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None:
                return None
            original = spec.loader

            class M0Loader(importlib.abc.Loader):
                def create_module(self, spec):
                    return original.create_module(spec)

                def exec_module(self, module):
                    original.exec_module(module)
                    if fullname == "sglang.srt.model_executor.model_runner":
                        from m0_observe import install
                        install(module.ModelRunner)
                    elif fullname == "sglang.srt.layers.attention.dsa.dsa_indexer":
                        from m0_observe import install_indexer
                        install_indexer(module.Indexer)
                    elif fullname == "sglang.srt.mem_cache.unified_radix_cache":  # [M1]
                        from m1_radix_observe import install as install_radix
                        install_radix(module.UnifiedRadixCache)
                    elif fullname == "sglang.srt.mem_cache.radix_cache":  # [M1] 兜底
                        from m1_radix_observe import install as install_radix
                        if hasattr(module, "RadixCache"):
                            install_radix(module.RadixCache)

            spec.loader = M0Loader()
            return spec

    sys.meta_path.insert(0, M0Finder())
