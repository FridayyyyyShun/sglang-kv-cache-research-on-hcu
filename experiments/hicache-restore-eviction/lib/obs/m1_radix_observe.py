"""M1 Radix Cache 观测钩子（drop-in，由 sitecustomize 在 import 时安装）。

现场实际用的是 UnifiedRadixCache（registry 默认），不是旧的 RadixCache。
二者接口差异很大，本钩子完全按 UnifiedRadixCache 写：

  * match_prefix(params) -> MatchResult(NamedTuple)；命中长度 = device_indices.numel()
  * insert(params) -> InsertResult(prefix_len, total_len, last_device_node, ...)
  * evict(params) -> EvictResult(num_tokens_evicted, ...)
  * inc_lock_ref(node_id, ...) / dec_lock_ref(node_id, params, ...)  —— 收 node_id（整数），不是 node
  * cache_finished_req(req, *, kv_len_to_handle) / cache_unfinished_req(req, chunked)
  * 没有 _delete_leaf；树在 self.tree_core.root_node，节点是 UnifiedTreeNode
      - node.children: dict[key -> UnifiedTreeNode]
      - node.key: RadixKey（len() = token 数）
      - node.component_data[FULL].value / .lock_ref   （FULL = ComponentType.FULL = 0）
      - node.id / node.hit_count / node.priority
  * 统计：cache.evictable_size() / cache.protected_size() / cache.total_size()
  * allocator：cache.token_to_kv_pool_allocator.available_size() / .free_pages

设计约束（别改成 M0 那样）：
  * 不调 torch.cuda.synchronize()：match/insert 是每请求路径，同步会掩盖真实现象。
  * 不进逐 token 热路径：只包裹对外方法。
  * 任何异常都吞掉，绝不影响服务。

产物（都落在 M0_TRACE_DIR）：
  radix_events_<pid>.jsonl     每请求/每次 evict/每次锁操作一条
  radix_snapshots_<pid>.jsonl  树快照（结构化，含父子、lock_ref、value 长度）
  触发快照：touch $M0_TRACE_DIR/snapshot.request   （跑完客户端后立刻取一次）
           或 M1_SNAPSHOT_EVERY=<n> 每 n 次 match_prefix 自动一次
"""

import functools
import hashlib
import json
import os
import time

_PID = os.getpid()
_FULL = 0  # ComponentType.FULL —— base component，Full-KV 的槽都挂在它上面

# SGLang 用 spawn 起 scheduler 子进程，M0_TRACE_DIR 可能丢失。
# 约定一个固定的 handoff 文件：谁先拿到 M0_TRACE_DIR 就写进去，后来的进程读它兜底。
_HANDOFF = os.environ.get("M1_TRACE_HANDOFF", "/tmp/m1-radix/.trace_dir")


def _dir():
    d = os.environ.get("M0_TRACE_DIR")
    if d:
        try:
            os.makedirs(os.path.dirname(_HANDOFF), exist_ok=True)
            with open(_HANDOFF, "w") as f:
                f.write(d)
        except Exception:
            pass
        return d
    try:
        with open(_HANDOFF) as f:
            d = f.read().strip()
        if d and os.path.isdir(d):
            return d
    except Exception:
        pass
    return "."


def _emit(name, d):
    d["pid"] = _PID
    d.setdefault("time", time.time())
    try:
        with open(os.path.join(_dir(), name), "a") as f:
            f.write(json.dumps(d, default=str) + "\n")
    except Exception as e:  # 观测失败不能影响推理
        print("M1_OBSERVER_ERROR", repr(e), flush=True)


def _page_size(cache):
    try:
        return int(cache.page_size) or 1
    except Exception:
        return 1


def _sizes(cache):
    """三个守恒量：evictable / protected / tokens(total_size)。"""
    out = {}
    try:
        out["evictable"] = int(cache.evictable_size())
    except Exception:
        pass
    try:
        out["protected"] = int(cache.protected_size())
    except Exception:
        pass
    try:
        ts = cache.total_size()
        out["tokens"] = int(ts[0]) if isinstance(ts, (tuple, list)) else int(ts)
    except Exception:
        pass
    return out


def _alloc_state(cache):
    al = getattr(cache, "token_to_kv_pool_allocator", None)
    out = {}
    if al is None:
        return out
    fn = getattr(al, "available_size", None)
    if callable(fn):
        try:
            out["available_tokens"] = int(fn())
        except Exception:
            pass
    fp = getattr(al, "free_pages", None)
    if fp is not None:
        try:
            out["free_pages"] = int(len(fp))
        except Exception:
            try:
                out["free_pages"] = int(fp.numel())
            except Exception:
                pass
    return out


def _key_meta(cache, key):
    ps = _page_size(cache)
    n = len(key) if key is not None else 0
    return dict(
        key_len=n,
        aligned_len=(n // ps) * ps,
        page_size=ps,
        extra_key=getattr(key, "extra_key", None),
        cache_salt=getattr(key, "cache_salt", None),
        is_bigram=getattr(key, "is_bigram", None),
    )


def _node_vlen(node):
    """FULL 组件在该节点上的槽数（= 该节点贡献的 token 数）。"""
    try:
        v = node.component_data[_FULL].value
        return int(len(v)) if v is not None else 0
    except Exception:
        return 0


def _node_lock(node):
    try:
        return int(node.component_data[_FULL].lock_ref)
    except Exception:
        return 0


def snapshot(cache, tag):
    """DFS 全树（走 tree_core.root_node），输出结构化节点表 + 守恒量。"""
    rows = []
    total = 0
    sum_lock = 0
    leaves = 0
    root = None
    try:
        root = cache.tree_core.root_node
    except Exception:
        root = getattr(cache, "root_node", None)
    if root is None:
        d = dict(event="snapshot", tag=tag, nodes=0, note="no_root")
        d.update(_sizes(cache))
        d.update(_alloc_state(cache))
        _emit("radix_snapshots_%d.jsonl" % _PID, d)
        return d

    cap = int(os.environ.get("M1_SNAPSHOT_NODE_CAP", "400000"))
    stack = [(root, 0, None)]
    while stack:
        node, depth, parent_id = stack.pop()
        if depth > 0:
            klen = len(node.key) if node.key is not None else 0
            vlen = _node_vlen(node)
            lock = _node_lock(node)
            total += vlen
            sum_lock += lock
            if len(node.children) == 0:
                leaves += 1
            if len(rows) < cap:
                rows.append(dict(id=node.id, parent=parent_id, depth=depth,
                                 klen=klen, lock=lock, hit=getattr(node, "hit_count", 0),
                                 pr=getattr(node, "priority", 0), val=vlen,
                                 host_v=getattr(node, "backuped", False)))
        for c in node.children.values():
            stack.append((c, depth + 1, node.id))

    d = dict(event="snapshot", tag=tag, nodes=len(rows), leaves=leaves,
             tokens_walk=total, sum_lock_ref=sum_lock,
             page_size=_page_size(cache),
             disable=getattr(cache, "disable", None),
             nodes_detail=rows)
    d.update(_sizes(cache))
    d.update(_alloc_state(cache))
    _emit("radix_snapshots_%d.jsonl" % _PID, d)
    return d


def _maybe_snapshot(cache):
    sentinel = os.path.join(_dir(), "snapshot.request")
    if os.path.exists(sentinel):
        tag = "manual"
        try:
            with open(sentinel) as f:
                tag = (f.read().strip() or "manual")
        except Exception:
            pass
        try:
            os.remove(sentinel)
        except Exception:
            pass
        snapshot(cache, tag)
        return
    every = int(os.environ.get("M1_SNAPSHOT_EVERY", "0") or 0)
    if every > 0:
        n = getattr(cache, "_m1_match_count", 0)
        if n and n % every == 0:
            snapshot(cache, "every%d" % every)


def _hc_node(cache, node_id):
    node = cache.tree_core.node_by_id(node_id)
    cd = node.component_data[_FULL]
    return dict(node_id=node_id,
                device_tokens=len(cd.value) if cd.value is not None else 0,
                host_tokens=len(cd.host_value) if cd.host_value is not None else 0,
                lock_ref=cd.lock_ref, host_lock_ref=cd.host_lock_ref)


def _verify_pages(cache, node, indices):
    """Diagnostic only: hash every byte, all layers, in logical page order.

    Use the pool's offload API so DSA index keys/scales are included too.
    Prefix-based identities survive radix node splits and physical slot changes.
    """
    import torch

    pool = cache.token_to_kv_pool_allocator.get_kvcache()
    if type(pool).__name__ not in ("MLATokenToKVPool", "DSATokenToKVPool"):
        raise RuntimeError("KV verification unsupported pool: " + type(pool).__name__)
    ps = _page_size(cache)
    n = len(indices)
    if n == 0 or n % ps or len(node.key) != n or node.key.is_bigram:
        raise RuntimeError("KV verification requires full, page-aligned non-bigram node")
    copies = pool.get_cpu_copy(indices)
    arrays = []

    def collect(value, path="root"):
        if isinstance(value, dict):
            for key in sorted(value):
                collect(value[key], path + "/" + key)
        elif isinstance(value, (list, tuple)):
            if value and all(isinstance(x, torch.Tensor) for x in value):
                arrays.append((path, torch.cat(value, dim=0)))
            else:
                for i, child in enumerate(value):
                    collect(child, path + "/" + str(i))
        else:
            raise RuntimeError("Unsupported KV CPU snapshot structure: " + path)

    collect(copies)
    if not arrays:
        raise RuntimeError("Empty KV snapshot")
    ancestors = []
    parent = node.parent
    while parent is not None and parent.parent is not None:
        ancestors.append(list(parent.key.token_ids)[:len(parent.key)])
        parent = parent.parent
    prefix = [t for part in reversed(ancestors) for t in part]
    tokens = list(node.key.token_ids)[:n]
    result = {}
    for page in range(n // ps):
        digest = hashlib.sha256()
        for path, tensor in arrays:
            if tensor.shape[0] == n:
                chunk = tensor[page * ps:(page + 1) * ps]
            elif tensor.shape[0] == n // ps:
                chunk = tensor[page:page + 1]
            else:
                raise RuntimeError("Unrecognized KV token/page axis: " + path)
            digest.update(str((path, str(chunk.dtype), tuple(chunk.shape))).encode())
            digest.update(chunk.contiguous().view(torch.uint2).numpy().tobytes())
        identity = hashlib.sha256(json.dumps(
            [node.key.cache_salt, prefix + tokens[:(page + 1) * ps]]
        ).encode()).hexdigest()
        result[identity] = digest.hexdigest()
    return result


def _verify_target(node):
    try:
        with open(os.path.join(_dir(), "kv_verify.request")) as f:
            salt = f.read().strip()
        return bool(salt) and node.key.cache_salt == salt
    except FileNotFoundError:
        return False


def _verify_backup(cache, nid, indices):
    node = cache.tree_core.node_by_id(nid)
    if not _verify_target(node):
        return
    pages = _verify_pages(cache, node, indices)
    saved = cache.__dict__.setdefault("_m1_verify_pages", {})
    # Do not overwrite the original baseline with data from a later restore.
    for key, digest in pages.items():
        if key in saved and saved[key] != digest:
            raise RuntimeError("KV changed before repeated backup: " + key)
        saved[key] = digest
    _emit("radix_events_%d.jsonl" % _PID,
          dict(event="hc_kv_baseline", node_id=nid, cache_salt=node.key.cache_salt,
               pages=len(pages), tokens=len(indices), algorithm="sha256"))


def _verify_loaded(cache, nid):
    path = cache.__dict__.get("_m1_verify_loads", {}).pop(nid, [])
    saved = cache.__dict__.get("_m1_verify_pages", {})
    checked = missing = mismatched = 0
    salt = None
    for loaded_id in path:
        node = cache.tree_core.node_by_id(loaded_id)
        if not _verify_target(node):
            continue
        salt = node.key.cache_salt
        pages = _verify_pages(cache, node, node.component_data[_FULL].value)
        for key, digest in pages.items():
            missing += key not in saved
            mismatched += key in saved and saved[key] != digest
            checked += 1
    return dict(event="hc_kv_verified", node_id=nid, cache_salt=salt,
                pages=checked, tokens=checked * _page_size(cache),
                missing_pages=missing, mismatched_pages=mismatched,
                success=checked > 0 and missing == 0 and mismatched == 0,
                algorithm="sha256")


def _install_hicache(cache_cls):
    # Capture queue state, but emit completions only after production code consumes
    # the corresponding ack. Default mode never copies/synchronizes KV. Explicit
    # M1_HICACHE_VERIFY=1 enables slow, target-only correctness diagnostics.
    names = ("_execute_kv_backup", "_finish_write_through_ack", "load_back", "loading_check")
    for name in names:
        orig = getattr(cache_cls, name, None)
        if orig is None:
            continue
        def make(orig, name):
            @functools.wraps(orig)
            def call(self, *a, **kw):
                before = {}
                try:
                    if os.environ.get("M1_HICACHE_VERIFY") == "1":
                        if name == "_execute_kv_backup":
                            nid = a[0] if a else kw["node_id"]
                            indices = a[1] if len(a) > 1 else kw["device_value"]
                            _verify_backup(self, nid, indices)
                        elif name == "load_back":
                            nid = a[0] if a else kw["node_id"]
                            node = self.tree_core.node_by_id(nid)
                            path = []
                            while node.parent is not None:
                                cd = node.component_data[_FULL]
                                if cd.value is None and cd.host_value is not None:
                                    path.append(node.id)
                                node = node.parent
                            self.__dict__.setdefault("_m1_verify_loads", {})[nid] = path
                    if name == "loading_check":
                        before = dict(self.ongoing_load_back)
                except Exception as exc:
                    _emit("radix_events_%d.jsonl" % _PID,
                          dict(event="hc_observer_error", method=name, error=repr(exc)))
                result = orig(self, *a, **kw)
                try:
                    rows = []
                    if name == "loading_check":
                        for nid in before:
                            if nid not in self.ongoing_load_back:
                                rows.append(dict(event="hc_load_done", **_hc_node(self, nid)))
                                if os.environ.get("M1_HICACHE_VERIFY") == "1":
                                    rows.append(_verify_loaded(self, nid))
                    else:
                        nid = a[0] if a else kw.get("node_id", kw.get("ack_id"))
                        row = _hc_node(self, nid)
                        row["event"] = {"_execute_kv_backup": "hc_backup_submit",
                                        "_finish_write_through_ack": "hc_backup_done",
                                        "load_back": "hc_load_submit"}[name]
                        row["success"] = result is not None if name == "_execute_kv_backup" else bool(result) if name == "load_back" else True
                        rows.append(row)
                    for row in rows:
                        row["rank"] = getattr(self, "pp_rank", None)
                        _emit("radix_events_%d.jsonl" % _PID, row)
                except Exception as exc:
                    _emit("radix_events_%d.jsonl" % _PID,
                          dict(event="hc_observer_error", method=name, error=repr(exc)))
                return result
            return call
        setattr(cache_cls, name, make(orig, name))


def install(cache_cls):
    if getattr(cache_cls, "_m1_installed", False):
        return
    cache_cls._m1_installed = True
    _install_hicache(cache_cls)

    # Opt-in T3 audit: capture eligible leaves before the production selector.
    orig_next = getattr(cache_cls, "_evict_device_next_node", None)
    if orig_next is not None:
        @functools.wraps(orig_next)
        def eviction_next(self, component_type, tracker, _orig=orig_next):
            candidates = None
            if os.environ.get("M1_EVICTION_VERIFY") == "1" and int(component_type) == _FULL:
                try:
                    core = self.tree_core
                    comp = core.components_by_type[component_type]
                    candidates = [dict(
                        node_id=n.id, last_access_time=n.last_access_time,
                        creation_time=n.creation_time, hit_count=n.hit_count,
                        priority=n.priority, session_ref=comp.session_ref(n),
                        lock_ref=n.component_data[_FULL].lock_ref,
                        cache_salt=getattr(n.key, "cache_salt", None))
                        for n in core.evictable_device_leaves]
                except Exception as exc:
                    _emit("radix_events_%d.jsonl" % _PID,
                          dict(event="eviction_observer_error", error=repr(exc)))
            result = _orig(self, component_type, tracker)
            if candidates is not None and result is not None:
                _emit("radix_events_%d.jsonl" % _PID, dict(
                    event="eviction_choice", selected=result, candidates=candidates,
                    strategy=type(self.tree_core.eviction_strategy).__name__))
            return result
        cache_cls._evict_device_next_node = eviction_next

    # ---------- 匹配 ----------
    orig = getattr(cache_cls, "match_prefix", None)
    if orig is not None:
        @functools.wraps(orig)
        def match_prefix(self, params, *a, _orig=orig, **kw):
            try:
                meta = _key_meta(self, getattr(params, "key", None))
            except Exception:
                meta = {}
            res = _orig(self, params, *a, **kw)
            try:
                di = getattr(res, "device_indices", None)
                hit = int(di.numel()) if di is not None else 0
                hhl = int(getattr(res, "host_hit_length", 0) or 0)
                d = dict(event="match", **meta, hit_tokens=hit, host_hit=hhl,
                         rid=str(getattr(getattr(params, "req", None), "rid", None)),
                         device_node=getattr(res, "last_device_node", None),
                         host_node=getattr(res, "last_host_node", None),
                         miss_tokens=max(0, meta.get("aligned_len", 0) - hit))
                d.update(_sizes(self))
                d.update(_alloc_state(self))
                _emit("radix_events_%d.jsonl" % _PID, d)
                self._m1_match_count = getattr(self, "_m1_match_count", 0) + 1
                _maybe_snapshot(self)
            except Exception:
                pass
            return res
        cache_cls.match_prefix = match_prefix

    # ---------- 插入 ----------
    orig = getattr(cache_cls, "insert", None)
    if orig is not None:
        @functools.wraps(orig)
        def insert(self, params, *a, _orig=orig, **kw):
            try:
                meta = _key_meta(self, getattr(params, "key", None))
                v = getattr(params, "value", None)
                meta["value_len"] = int(len(v)) if v is not None else None
            except Exception:
                meta = {}
            res = _orig(self, params, *a, **kw)
            try:
                d = dict(event="insert", **meta,
                         node_id=getattr(res, "last_device_node", None),
                         prefix_len=getattr(res, "prefix_len", None),
                         total_len=getattr(res, "total_len", None))
                d.update(_sizes(self))
                d.update(_alloc_state(self))
                _emit("radix_events_%d.jsonl" % _PID, d)
            except Exception:
                pass
            return res
        cache_cls.insert = insert

    # ---------- 驱逐 ----------
    orig = getattr(cache_cls, "evict", None)
    if orig is not None:
        @functools.wraps(orig)
        def evict(self, params, *a, _orig=orig, **kw):
            started = time.perf_counter()
            before = None
            try:
                before = dict(**_sizes(self), **_alloc_state(self))
            except Exception:
                pass
            res = _orig(self, params, *a, **kw)
            duration_ms = (time.perf_counter() - started) * 1000
            try:
                d = dict(event="evict",
                         duration_ms=duration_ms,
                         requested=getattr(params, "num_tokens", None),
                         evicted=getattr(res, "num_tokens_evicted", None),
                         before=before,
                         after=dict(**_sizes(self), **_alloc_state(self)))
                _emit("radix_events_%d.jsonl" % _PID, d)
            except Exception:
                pass
            return res
        cache_cls.evict = evict

    # ---------- 锁引用：收 node_id（整数），需要解析回节点取 klen ----------
    def _node_klen(self, node_id):
        try:
            node = self.tree_core.node_by_id(node_id)
            return len(node.key) if node.key is not None else 0, _node_lock(node)
        except Exception:
            return None, None

    for name, sign in (("inc_lock_ref", +1), ("dec_lock_ref", -1)):
        orig = getattr(cache_cls, name, None)
        if orig is None:
            continue

        def make(orig, name, sign):
            @functools.wraps(orig)
            def call(self, node_id, *a, **kw):
                res = orig(self, node_id, *a, **kw)
                try:
                    klen, lock = _node_klen(self, node_id)
                    d = dict(event=name, sign=sign, node_id=node_id,
                             klen=klen, node_lock=lock,
                             delta=getattr(res, "delta", None))
                    d.update(_sizes(self))
                    _emit("radix_events_%d.jsonl" % _PID, d)
                except Exception:
                    pass
                return res
            return call
        setattr(cache_cls, name, make(orig, name, sign))

    # ---------- 请求生命周期 ----------
    for name in ("cache_finished_req", "cache_unfinished_req"):
        orig = getattr(cache_cls, name, None)
        if orig is None:
            continue

        def make(orig, name):
            @functools.wraps(orig)
            def call(self, req, *a, **kw):
                res = orig(self, req, *a, **kw)
                try:
                    fid = getattr(req, "fill_ids", None)
                    d = dict(event=name,
                             rid=str(getattr(req, "rid", None)),
                             fill_len=(len(fid) if fid is not None else None),
                             cached=getattr(req, "cached_tokens", None),
                             kv_len=kw.get("kv_len_to_handle"),
                             chunked=bool(kw.get("chunked", False)))
                    d.update(_sizes(self))
                    d.update(_alloc_state(self))
                    _emit("radix_events_%d.jsonl" % _PID, d)
                except Exception:
                    pass
                return res
            return call
        setattr(cache_cls, name, make(orig, name))

    # ---------- reset / flush（T4 恢复用） ----------
    orig = getattr(cache_cls, "reset", None)
    if orig is not None:
        @functools.wraps(orig)
        def reset(self, *a, _orig=orig, **kw):
            res = _orig(self, *a, **kw)
            try:
                d = dict(event="reset")
                d.update(_sizes(self))
                d.update(_alloc_state(self))
                _emit("radix_events_%d.jsonl" % _PID, d)
            except Exception:
                pass
            return res
        cache_cls.reset = reset

    print("M1_RADIX_OBSERVE installed on", cache_cls.__name__, flush=True)
