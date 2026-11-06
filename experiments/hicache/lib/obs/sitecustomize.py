"""Process-local observer bootstrap; also suppresses the historical M1 site hook.

The existing M1 .pth imports sitecustomize by module name. With this directory
first on PYTHONPATH it resolves here, so no M1 handoff/snapshots are touched.
"""
import hicache_bootstrap  # noqa: F401
