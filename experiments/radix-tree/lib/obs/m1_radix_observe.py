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
import json
import os
import time

_PID = os.getpid()
_FULL = 0  # ComponentType.FULL —— base component，Full-KV 的槽都挂在它上面

# SGLang 用 spawn 起 scheduler 子进程，M0_TRACE_DIR 可能丢失。
# 约定一个固定的 handoff 文件：谁先拿到 M0_TRACE_DIR 就写进去，后来的进程读它兜底。
_HANDOFF = "/tmp/m1-radix/.trace_dir"


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


def install(cache_cls):
    if getattr(cache_cls, "_m1_installed", False):
        return
    cache_cls._m1_installed = True

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
            before = None
            try:
                before = dict(**_sizes(self), **_alloc_state(self))
            except Exception:
                pass
            res = _orig(self, params, *a, **kw)
            try:
                d = dict(event="evict",
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
