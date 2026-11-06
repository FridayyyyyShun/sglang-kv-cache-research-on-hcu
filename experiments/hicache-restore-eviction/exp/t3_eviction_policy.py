# Run from the experiment root (parent of exp/), with local lib/env.sh configured.
# Terminal 1: start the server; wait until ready.
# M1_HICACHE_VERIFY=0 M1_EVICTION_VERIFY=1 bash lib/serve.sh \
#   --hicache on --hc-ratio 4 --hc-policy write_through \
#   --hc-layout page_first --hc-io kernel --policy lru \
#   --radix on --kv bf16 --page 64 --cap 16384 --ctx 32768 --chunk 8192 \
#   --port 31000 --src repo --obs 1 --root /workspace/kv-runtime
# Terminal 2: run the test (replace SERVER_IP with the actual address).
# bash run_hicache.sh --only t3 --host SERVER_IP --policy lru
# The runner reads .state/last_run; use --run-dir to override.
# Restart the server for each policy; server and client --policy must agree.

"""T3: audit real GPU eviction choices against independently computed policy keys.

Start serve.sh with M1_EVICTION_VERIFY=1 and --policy lru|lfu|slru|priority.
No flush, generated-answer comparison, or inference from latency is performed.
"""
import json
from pathlib import Path
import sys
import uuid
import time
import math

import common as C
import hicache_common as H


STRATEGIES = dict(lru='LRUStrategy', lfu='LFUStrategy',
                  slru='SLRUStrategy', priority='PriorityStrategy')


def policy_key(node, policy):
    last = node['last_access_time']
    key = {'lru': (last,), 'lfu': (node['hit_count'], last),
           'slru': (int(node['hit_count'] >= 2), last),
           'priority': (node['priority'], last)}[policy]
    ref = node['session_ref']
    return (ref > 0, ref) + key


def audit(event, policy):
    candidates = event['candidates']
    if event['strategy'] != STRATEGIES[policy] or not candidates:
        return False
    selected = [n for n in candidates if n['node_id'] == event['selected']]
    return (len(selected) == 1 and selected[0]['lock_ref'] == 0
            and policy_key(selected[0], policy) == min(policy_key(n, policy) for n in candidates))


def performance_summary(requests):
    latencies = sorted(x['latency_ms'] for x in requests)
    seconds = sum(latencies) / 1000
    def percentile(q):
        return latencies[max(0, math.ceil(len(latencies) * q) - 1)] if latencies else None
    return dict(requests=len(requests), request_time_s=round(seconds, 6),
                latency_mean_ms=sum(latencies) / len(latencies) if latencies else None,
                latency_p50_ms=percentile(.5), latency_p95_ms=percentile(.95),
                request_rate_rps=len(requests) / seconds if seconds else None,
                input_tokens_per_s=sum(x['input_tokens'] for x in requests) / seconds if seconds else None,
                cached_prompt_fraction=(sum(x['cached_tokens'] for x in requests)
                    / sum(x['input_tokens'] for x in requests))
                    if requests and all(x['cached_tokens'] is not None for x in requests) else None)


def main():
    p = H.parser(__doc__)
    p.add_argument('--policy', choices=STRATEGIES, default='lru')
    p.add_argument('--pressure-len', type=int, default=8192)
    p.add_argument('--max-rounds', type=int, default=6)
    p.add_argument('--expected-ranks', type=int, default=8)
    a = C.setup(p.parse_args(), 't3_eviction_policy')
    if min(a.pressure_len, a.max_rounds, a.expected_ranks) <= 0:
        p.error('length, rounds and ranks must be positive')
    r = C.Report(a, 't3_eviction_policy')
    start = C.mark_events(a)
    evidence = {'config': vars(a).copy(), 'requests': []}
    def generate(stage, tokens, namespace):
        begin = time.perf_counter()
        result = C.gen(a, tokens, salt=namespace)
        elapsed = time.perf_counter() - begin
        row = dict(kind='request', stage=stage, input_tokens=len(tokens),
                   latency_ms=round(elapsed * 1000, 3), cached_tokens=result['cached'],
                   completion_tokens=result['raw'].get('completion_tokens'))
        evidence['requests'].append(row)
        r.row(**row)
        print('[T3] %s input=%d latency=%.2f ms cached=%s' %
              (stage, len(tokens), elapsed * 1000, result['cached']), flush=True)
    try:
        salt = H.prepare(a) + '-t3-' + uuid.uuid4().hex
        # Unequal frequency plus a final recency refresh exercises ordering.
        for index in [0, 1, 0, 0, 2, 1]:
            generate('target-%d' % index, C.ids(a.prompt_len, seed=711 + index), salt + '-' + str(index))
        for i in range(a.max_rounds):
            generate('pressure-%d' % (i + 1), C.ids(a.pressure_len, seed=1900 + i), salt + '-pressure-' + str(i))
            choices = C.events_since(a, start, 'eviction_choice')
            informative = [e for e in choices if len(e.get('candidates', [])) >= 2]
            if len({e['pid'] for e in informative}) >= a.expected_ranks:
                break
        es = C.events_since(a, start)
        choices = [e for e in es if e.get('event') == 'eviction_choice']
        informative = [e for e in choices if len(e.get('candidates', [])) >= 2]
        pids = {e['pid'] for e in informative}
        r.check(len(pids) == a.expected_ranks, 'multi-candidate eviction observed on all ranks', sorted(pids))
        bad = [e for e in choices if not audit(e, a.policy)]
        r.check(bool(choices) and not bad, 'selected victims follow configured policy and are unlocked',
                dict(choices=len(choices), failures=len(bad)))
        for pid in sorted(pids):
            own = [e for e in choices if e['pid'] == pid]
            evictions = [e for e in es if e.get('event') == 'evict' and e['pid'] == pid]
            durations = [e['duration_ms'] for e in evictions if 'duration_ms' in e]
            row = dict(kind='rank', pid=pid, policy=a.policy, choices=len(own),
                       failures=sum(not audit(e, a.policy) for e in own),
                       eviction_calls=len(evictions),
                       evicted_tokens=sum(e.get('evicted', 0) or 0 for e in evictions),
                       eviction_total_ms=sum(durations) if durations else None,
                       eviction_mean_ms=sum(durations) / len(durations) if durations else None)
            r.row(**row)
            print('[T3] rank metrics: ' + json.dumps(row), flush=True)
    except Exception as exc:
        r.check(False, 'experiment completed without exception', repr(exc))
    finally:
        es = C.events_since(a, start)
        evidence['events'] = es
        metrics = performance_summary(evidence['requests'])
        evidence['performance'] = metrics
        r.row(kind='summary', **metrics)
        print('[T3] performance (audit enabled): ' + json.dumps(metrics), flush=True)
        r.check(not [e for e in es if e.get('event') in ('eviction_observer_error', 'hc_observer_error')],
                'observer has no errors')
        Path(a.out + '.artifacts.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    return r.finish()


if __name__ == '__main__':
    sys.exit(main())
