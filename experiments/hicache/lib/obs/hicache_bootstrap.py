"""Lazy observer installation; never import SGLang during site initialization."""
import importlib.abc
import importlib.machinery
import os
import sys

if os.environ.get("HICACHE_OBSERVE") == "1":
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in ('sglang.srt.mem_cache.unified_radix_cache', 'sglang.srt.managers.scheduler'):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None:
                return None
            original = spec.loader

            class Loader(importlib.abc.Loader):
                def create_module(self, spec):
                    return original.create_module(spec)

                def exec_module(self, module):
                    original.exec_module(module)
                    from hicache_observe import install, install_scheduler
                    if fullname == 'sglang.srt.managers.scheduler':
                        install_scheduler(module.Scheduler)
                    else:
                        install(module.UnifiedRadixCache)

            spec.loader = Loader()
            return spec

    sys.meta_path.insert(0, Finder())
