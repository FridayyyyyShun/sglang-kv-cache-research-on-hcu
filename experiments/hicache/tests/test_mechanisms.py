#!/usr/bin/env python3
"""Mechanism fixture tests; no GPU, no performance measurement, no SGLang import."""
import copy
import contextlib
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'exp'),str(ROOT/'lib/obs')]
from common import (Experiment, NotExercised, abort_reason, aligned, baseline_signature, branching,
                    invariants, parser, state_of)
import hicache_observe as obs


class Tensor:
    def __init__(self,n): self.n=n
    def numel(self): return self.n


def fixture(tp=0):
    def cd(n,h,dl=0,hl=0): return NS(value=Tensor(n) if n else None,
        host_value=Tensor(h) if h else None,lock_ref=dl,host_lock_ref=hl)
    root=NS(id=0,parent=None,children={})
    parent=root
    for nid,key in ((1,[10]*64),(2,[20]*64)):
        node=NS(id=nid,parent=parent,children={},key=NS(token_ids=key,cache_salt=None),
            component_data=[cd(64 if nid==1 else 0,64)],write_through_pending_id=None,
            load_back_pending_id=None,hash_value=['hash'+str(nid)])
        # RadixKey exposes length; use a small real value object below.
        class Key:
            cache_salt=None
            def __init__(self,value): self.token_ids=value
            def __len__(self): return len(self.token_ids)
        node.key=Key(key); parent.children[nid]=node; parent=node
    cache=NS(tree_core=NS(root_node=root),tree_components=[0],cache_controller=object(),
        enable_storage=True,tp_rank=tp,pp_rank=0,page_size=64,host_memory_mode='cache',
        host_pool_group=NS(entry_map={
            'kv':NS(host_pool=NS(available_size=lambda:896,logical_size=1024)),
            'indexer':NS(host_pool=NS(available_size=lambda:896,logical_size=1024))}),
        total_size=lambda:(64,0),evictable_size=lambda:64,protected_size=lambda:0,
        token_to_kv_pool_allocator=NS(available_size=lambda:960),
        ongoing_write_through={},ongoing_load_back={},ongoing_prefetch={},ongoing_backup={})
    class UnifiedRadixCache: pass
    actual=UnifiedRadixCache(); actual.__dict__.update(vars(cache))
    del actual.tp_rank
    scheduler=NS(is_fully_idle=lambda:True,running_batch=NS(reqs=[]),waiting_queue=[],ps=NS(tp_rank=tp,pp_rank=0))
    return actual,scheduler


class Handler(BaseHTTPRequestHandler):
    abort=False
    def log_message(self,*args): pass
    def do_POST(self):
        payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.send_response(200);self.end_headers()
        row={'output_ids':[198],'text':'x','meta_info':{'cached_tokens':64,
            'cached_tokens_details':{'device':0,'host':0,'storage':64,'storage_backend':'file'},
            'completion_tokens':1,'finish_reason':{'type':'abort' if self.abort else 'length'}}}
        self.wfile.write(('data: '+json.dumps(row)+'\n\ndata: [DONE]\n\n').encode())


class MechanismTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.run=Path(self.temp.name)
        self.old=os.environ.get('HICACHE_TRACE_DIR');os.environ['HICACHE_TRACE_DIR']=str(self.run)
    def tearDown(self):
        if self.old is None:os.environ.pop('HICACHE_TRACE_DIR',None)
        else:os.environ['HICACHE_TRACE_DIR']=self.old
        self.temp.cleanup()
    def sample(self,tp=0):
        cache,sched=fixture(tp)
        # snapshot reports actual class name; fixture should emulate that facade.
        cache.__class__ if False else None
        return obs.snapshot(cache,sched,'tag')
    def experiment(self,port=32000,ranks=1):
        (self.run/'axes.env').write_text('host=127.0.0.1\nport=%d\npage=64\ncap=8192\nobs=1\ntier=storage\n'%port)
        args=parser().parse_args(['--host','127.0.0.1','--port',str(port),'--run-dir',str(self.run),'--ranks',str(ranks),'--timeout','.2'])
        return Experiment(args,'test')
    def test_snapshot_distinguishes_host_only_and_sidecar_pools(self):
        sample=self.sample()
        invariants(sample)
        self.assertEqual(state_of(sample,[10]*64+[20]*64),dict(matched=128,device=64,host=128))
        self.assertEqual(set(sample['host_pools']),{'kv','indexer'})
        self.assertEqual(sample['nodes'][1]['components']['0']['device_slots'],0)
    def test_salt_namespace_does_not_match_different_tree(self):
        sample=self.sample(); sample['nodes'][0]['salt']='A'
        self.assertEqual(state_of(sample,[10]*64,'B')['matched'],0)
        self.assertEqual(state_of(sample,[10]*64,'A')['matched'],64)
    def test_root_omitted_and_page_sum_and_lock_violations_detected(self):
        sample=self.sample()
        self.assertEqual(len(sample['nodes']),2)
        sample['nodes'][0]['components']['0']['host_lock']=-1
        with self.assertRaises(AssertionError):invariants(sample)
        sample=self.sample();sample['device_tree_tokens']+=64
        with self.assertRaises(AssertionError):invariants(sample)
    def test_flush_empty_requires_tree_and_async_queues_empty(self):
        sample=self.sample()
        with self.assertRaises(AssertionError):invariants(sample,empty=True)
        sample.update(nodes=[],device_tree_tokens=0,evictable=0,protected=0)
        invariants(sample,empty=True)
        sample['ongoing']['ongoing_prefetch']=1
        with self.assertRaises(AssertionError):invariants(sample,empty=True)
    def test_snapshot_mailbox_once_per_tag_without_generation(self):
        cache,sched=fixture()
        class Scheduler:
            def process_input_requests(self,*args):return 'unchanged'
            def is_fully_idle(self):return True
        s=Scheduler();s.tree_cache=cache;s.running_batch=sched.running_batch;s.waiting_queue=[];s.ps=sched.ps
        obs.install_scheduler(Scheduler)
        (self.run/'snapshot.request.json').write_text(json.dumps({'tag':'one'}))
        for i in range(3):self.assertEqual(s.process_input_requests([]),'unchanged')
        entries=[json.loads(l) for p in self.run.glob('hicache_snapshots*') for l in p.read_text().splitlines()]
        self.assertEqual(len(entries),1)
        self.assertEqual(entries[0]['tag'],'one')
    def test_snapshot_mailbox_errors_do_not_break_scheduler(self):
        class Scheduler:
            def process_input_requests(self,*args):return 42
        obs.install_scheduler(Scheduler)
        (self.run/'snapshot.request.json').write_text('{incomplete')
        self.assertEqual(Scheduler().process_input_requests([]),42)
    def test_host_only_precondition_cannot_pass_with_device_hit(self):
        e=self.experiment();e.a.pressure_limit=2
        sample=self.sample()
        e.idle=lambda label:[sample]
        e.gen=lambda *args:None
        with self.assertRaises(NotExercised):e.to_host([10]*64+[20]*64+[30])
    def test_cancellation_requires_abort_terminal_not_http_ack(self):
        self.assertFalse(abort_reason({'meta_info':{'finish_reason':{'type':'length'}}}))
        self.assertTrue(abort_reason({'meta_info':{'finish_reason':{'type':'abort'}}}))
    def test_pool_baseline_includes_all_host_auxiliary_pools(self):
        sample=self.sample(); other=copy.deepcopy(sample)
        other['host_pools']['indexer']['available']-=1
        self.assertNotEqual(baseline_signature([sample]),baseline_signature([other]))
    def test_native_sse_retains_l3_evidence_and_has_no_performance_fields(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            e=self.experiment(server.server_port)
            row=e.gen([10]*65,'storage_fixture')
            self.assertEqual(row['details']['storage'],64)
            self.assertEqual(row['details']['storage_backend'],'file')
            self.assertNotIn('ttft_ms',row);self.assertNotIn('wall_ms',row)
            e.expect_hit(row,64,'storage')
            with self.assertRaises(AssertionError):e.expect_hit(row,64,'host')
        finally:server.shutdown();server.server_close()
    def test_positive_l3_response_without_transfer_evidence_is_not_pass(self):
        e=self.experiment()
        e.result['rows'].append({'req_id':'fake_storage','details':{'device':0,'host':0,'storage':64}})
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(e.finish('PASS'),1)
        self.assertEqual(e.result['status'],'FAIL')

    def test_layer_accounting_rejects_inconsistent_response(self):
        e=self.experiment()
        with self.assertRaises(AssertionError):e.check(64==128,'inconsistent details')
    def test_lazy_bootstrap_has_no_early_model_import(self):
        env=dict(os.environ,PYTHONPATH=str(ROOT/'lib/obs'),HICACHE_OBSERVE='1',PYTHONDONTWRITEBYTECODE='1')
        result=subprocess.run([sys.executable,'-B','-c','import sys; assert "sglang" not in sys.modules; import hicache_bootstrap; print("ok")'],env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
    def test_all_rank_snapshots_use_scheduler_parallel_state(self):
        e=self.experiment(ranks=2); e.a.timeout=2
        stopped=threading.Event()
        def respond():
            while not stopped.wait(.01):
                path=self.run/'snapshot.request.json'
                if path.exists():
                    command=json.loads(path.read_text())
                    for tp in (0,1):
                        cache,scheduler=fixture(tp)
                        obs.snapshot(cache,scheduler,command['tag'])
                    return
        thread=threading.Thread(target=respond,daemon=True);thread.start()
        try:
            values=e.snaps('two_ranks')
            self.assertEqual({s['tp_rank'] for s in values},{0,1})
        finally:stopped.set();thread.join(timeout=2)

    def test_missing_rank_snapshots_fail_instead_of_claiming_pass(self):
        e=self.experiment(ranks=2)
        with self.assertRaises(RuntimeError):e.snaps('missing')
    def test_branching_is_cumulative_prefix_not_node_key_length(self):
        sample=self.sample();parent=sample['nodes'][0]
        parent['children']=[2,3]
        self.assertTrue(branching(sample,[10]*64))
        self.assertFalse(branching(sample,[20]*64))


if __name__=='__main__':unittest.main()
