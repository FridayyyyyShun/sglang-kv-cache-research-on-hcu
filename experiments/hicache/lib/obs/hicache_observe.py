"""M1–M6 read-only Unified HiCache evidence. No tensors copied or device syncs.
Snapshots are taken in Scheduler.process_input_requests, including while idle.
Only a uniquely tagged snapshot mailbox is read; it never mutates cache state.
"""
import functools
import json
import os
from pathlib import Path
import time


def emit(event, **data):
    try:
        root = os.environ.get('HICACHE_TRACE_DIR')
        if not root:
            return
        data.update(event=event, time=time.time(), monotonic_ns=time.monotonic_ns(), pid=os.getpid())
        with (Path(root)/('hicache_events_%d.jsonl' % os.getpid())).open('a') as f:
            f.write(json.dumps(data, default=str)+'\n')
    except Exception:
        pass


def size(value):
    return 0 if value is None else int(value.numel())


def nodes(cache):
    stack = [cache.tree_core.root_node]
    while stack:
        node = stack.pop()
        if node.parent is not None:
            yield node
        stack.extend(node.children.values())


def protected(cache, host=False):
    result = {}
    for node in nodes(cache):
        cd = node.component_data[0]
        lock = cd.host_lock_ref if host else cd.lock_ref
        if lock > 0:
            result[node.id] = size(cd.host_value if host else cd.value)
    return result


def snapshot(cache, scheduler, tag):
    rows = []
    for node in nodes(cache):
        components = {}
        for ct in cache.tree_components:
            cd = node.component_data[int(ct)]
            components[str(int(ct))] = dict(device_slots=size(cd.value), host_slots=size(cd.host_value),
                device_lock=int(cd.lock_ref), host_lock=int(cd.host_lock_ref))
        rows.append(dict(id=node.id, parent=node.parent.id, key_len=len(node.key),
            key_tokens=list(node.key.token_ids), salt=getattr(node.key,'cache_salt',None),
            children=list(c.id for c in node.children.values()), components=components,
            write_pending=node.write_through_pending_id, load_pending=node.load_back_pending_id,
            hash_values=list(node.hash_value or [])))
    pools = {}
    group = getattr(cache,'host_pool_group',None)
    if group is not None:
        for name, entry in group.entry_map.items():
            pool=entry.host_pool
            pools[str(name)]={'available':int(pool.available_size()),
                              'logical_size':int(pool.logical_size)}
    total=cache.total_size()
    allocator=cache.token_to_kv_pool_allocator
    ongoing = {name:len(getattr(cache,name,{})) for name in
        ('ongoing_write_through','ongoing_load_back','ongoing_prefetch','ongoing_backup')}
    controller=cache.cache_controller
    queues={}
    for name in ('ack_write_queue','ack_load_queue','ack_backup_queue','ack_prefetch_queue',
                 'prefetch_hit_queue','host_mem_release_queue'):
        queue=getattr(controller,name,None)
        queues[name]=(int(queue.qsize()) if hasattr(queue,'qsize') else len(queue) if queue is not None else 0)
    for name,queue in getattr(controller,'extra_host_mem_release_queues',{}).items():
        queues['extra_release_'+str(name)]=int(queue.qsize())
    result=dict(event='snapshot', tag=tag, pid=os.getpid(), time=time.time(),
        tp_rank=int(scheduler.ps.tp_rank), pp_rank=int(scheduler.ps.pp_rank),
        cache_class=type(cache).__name__, hicache_attached=cache.cache_controller is not None,
        enable_storage=bool(cache.enable_storage), page_size=int(cache.page_size),
        device_tree_tokens=int(total[0] if isinstance(total,(tuple,list)) else total),
        evictable=int(cache.evictable_size()), protected=int(cache.protected_size()),
        device_available=int(allocator.available_size()),host_pools=pools,ongoing=ongoing,
        control_queues=queues,prefetch_tokens_occupied=int(getattr(controller,'prefetch_tokens_occupied',0)),
        fully_idle=bool(scheduler.is_fully_idle()),
        running=len(scheduler.running_batch.reqs), waiting=len(scheduler.waiting_queue),nodes=rows)
    path=Path(os.environ['HICACHE_TRACE_DIR'])/('hicache_snapshots_%d.jsonl'%os.getpid())
    with path.open('a') as f:
        f.write(json.dumps(result,default=str)+'\n')
    return result


def install_scheduler(cls):
    if cls.__dict__.get('_hicache_snapshot_hook',False): return
    cls._hicache_snapshot_hook=True
    original=cls.process_input_requests
    @functools.wraps(original)
    def process(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        try:
            root=os.environ.get('HICACHE_TRACE_DIR')
            if not root: return result
            path=Path(root)/'snapshot.request.json'
            if not path.exists(): return result
            command=json.loads(path.read_text())
            tag=command['tag']
            if getattr(self,'_hicache_snapshot_tag',None)==tag: return result
            snapshot(self.tree_cache,self,tag)
            self._hicache_snapshot_tag=tag
        except Exception as exc:
            emit('snapshot_error',error=repr(exc))
        return result
    cls.process_input_requests=process


def install(cls):
    if cls.__dict__.get('_hicache_observer',False): return
    cls._hicache_observer=True
    def wrap(name, original):
        @functools.wraps(original)
        def call(self,*args,**kwargs):
            pre={}
            try:
                if name in ('evict','evict_host'):
                    pre=protected(self,host=name=='evict_host')
                elif name in ('writing_check','loading_check','_drain_storage_control_queues_impl'):
                    field={'writing_check':'ongoing_write_through','loading_check':'ongoing_load_back',
                           '_drain_storage_control_queues_impl':'ongoing_backup'}[name]
                    pre=dict(getattr(self,field,{}))
                elif name in ('release_aborted_request','revoke_pending_prefetch'):
                    # 调用后条目已被删除，必须在调用前记下 abort 落地时预取是否仍在进行
                    rid=args[0] if args else kwargs.get('req_id',kwargs.get('rid'))
                    pre={'ongoing_before':rid in getattr(self,'ongoing_prefetch',{})}
            except Exception:
                pass
            result=original(self,*args,**kwargs)
            try:
                first=args[0] if args else kwargs.get('params')
                data={}
                if name=='match_prefix':
                    data.update(req_id=getattr(getattr(first,'req',None),'rid',None),
                        device_hit_tokens=int(result.device_indices.numel()),host_hit_tokens=int(result.host_hit_length),
                        key_len=len(first.key), last_device_node=result.last_device_node,
                        last_host_node=result.last_host_node,best_match_node=result.best_match_node)
                elif name=='insert':
                    data.update(key_len=len(first.key),prefix_len=result.prefix_len)
                elif name=='init_hicache':
                    data.update(cache_class=type(self).__name__,enable_storage=bool(self.enable_storage))
                elif name=='init_load_back':
                    data.update(req_id=getattr(first.req,'rid',None),host_hit_tokens=first.host_hit_length,
                                loaded_device_tokens=int(result[0].numel()))
                elif name=='_handle_prefetch_result':
                    data.update(req_id=first.request_id,storage_loaded_tokens=int(
                        self.prefetch_loaded_tokens_by_reqid.get(first.request_id,0)))
                elif name in ('prefetch_from_storage','release_aborted_request','revoke_pending_prefetch'):
                    request_id=args[0] if args else kwargs.get('req_id',kwargs.get('rid'))
                    data.update(req_id=request_id,ongoing=request_id in self.ongoing_prefetch,**pre)
                elif name in ('cache_finished_req','cache_unfinished_req'):
                    reason=getattr(first,'finished_reason',None)
                    data.update(req_id=getattr(first,'rid',None),finish_reason=(reason.to_json() if reason else None))
                elif name in ('evict','evict_host'):
                    host=name=='evict_host'
                    current={n.id:size(n.component_data[0].host_value if host else n.component_data[0].value)
                             for n in nodes(self)}
                    violations={str(nid):[n,current.get(nid,0)] for nid,n in pre.items() if current.get(nid,0)<n}
                    data.update(protected_before=pre,locked_page_violations=violations,
                        evicted_tokens=(int(result) if host else result.num_tokens_evicted))
                elif name in ('inc_lock_ref','dec_lock_ref','inc_host_lock_ref','dec_host_lock_ref'):
                    node_id=args[0] if args else kwargs.get('node_id')
                    node=self.resolve_node_handle(node_id)
                    data.update(node_id=node_id,device_lock=int(node.component_data[0].lock_ref),
                                host_lock=int(node.component_data[0].host_lock_ref))
                elif name in ('writing_check','loading_check','_drain_storage_control_queues_impl'):
                    field={'writing_check':'ongoing_write_through','loading_check':'ongoing_load_back',
                           '_drain_storage_control_queues_impl':'ongoing_backup'}[name]
                    done=[key for key in pre if key not in getattr(self,field,{})]
                    if not done: return result
                    data.update(completed_ids=done)
                emit(name,**data)
            except Exception as exc:
                emit('observer_error',method=name,error=repr(exc))
            return result
        return call
    names=('init_hicache','match_prefix','insert','init_load_back','_handle_prefetch_result',
        'prefetch_from_storage','release_aborted_request','revoke_pending_prefetch',
        'cache_finished_req','cache_unfinished_req','evict','evict_host','inc_lock_ref','dec_lock_ref',
        'inc_host_lock_ref','dec_host_lock_ref','writing_check','loading_check','_drain_storage_control_queues_impl')
    for name in names:
        original=getattr(cls,name,None)
        if original is not None: setattr(cls,name,wrap(name,original))
    emit('observer_installed',cache_class=cls.__name__)
