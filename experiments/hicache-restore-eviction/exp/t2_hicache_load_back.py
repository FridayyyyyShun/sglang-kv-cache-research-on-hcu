# Run from the experiment root (parent of exp/), with local lib/env.sh configured.
# Terminal 1: start the server; wait until ready.
# M1_HICACHE_VERIFY=1 M1_EVICTION_VERIFY=0 bash lib/serve.sh \
#   --hicache on --hc-ratio 4 --hc-policy write_through \
#   --hc-layout page_first --hc-io kernel --policy lru \
#   --radix on --kv bf16 --page 64 --cap 16384 --ctx 32768 --chunk 8192 \
#   --port 31000 --src repo --obs 1 --root /workspace/kv-runtime
# Terminal 2: run the test (replace SERVER_IP with the actual address).
# bash run_hicache.sh --only t2 --host SERVER_IP
# The runner reads .state/last_run; use --run-dir to override.
# Verifies all-rank restore and KV fingerprints; not a latency benchmark.

"""T2: all-rank GPU eviction, Host restore, and original/restored KV equality.

Use a small --cap and a larger --hc-ratio. A failed scenario is reported as
failure, never inferred to pass from latency or cached_tokens alone.
Restart the updated observer with M1_HICACHE_VERIFY=1. Verification synchronizes
GPU work and copies target KV plus DSA index cache to CPU for SHA-256 hashing.
This is a correctness diagnostic, not a latency benchmark or output-text test.
"""
import sys
import json
from pathlib import Path
import common as C
import hicache_common as H


def first_matches(events, salt):
    first = {}
    for e in sorted(events, key=lambda e: e.get('time', 0)):
        if e.get('event') == 'match' and e.get('cache_salt') == salt:
            first.setdefault(e['pid'], e)
    return first


def pressure_is_cold(first, expected):
    rids = {e.get('rid') for e in first.values()}
    return (set(first) == expected and len(rids) == 1
            and next(iter(rids), None) not in (None, 'None', '')
            and all(e.get('hit_tokens') == 0 and e.get('host_hit') == 0
                    for e in first.values()))


def record_request(r, evidence, stage, round_index, request_index, length, response, first):
    gpu = [e.get('hit_tokens') for e in first.values()]
    host = [e.get('host_hit') for e in first.values()]
    row = dict(stage=stage, round=round_index, request_index=request_index,
               input_tokens=length, cached=response['cached'], wall=response['wall'],
               observed_ranks=len(first),
               gpu_hit=max(gpu) if gpu and all(x is not None for x in gpu) else None,
               host_hit=max(host) if host and all(x is not None for x in host) else None)
    r.row(**row)
    evidence.setdefault('requests', []).append(dict(**row, first_matches=list(first.values())))
    print('[T2] ' + json.dumps(row), flush=True)


def verified_restore(events, salt, expected, minimum):
    submitted = {(e['pid'], e['node_id']) for e in events
                 if e.get('event') == 'hc_load_submit' and e.get('success')}
    done = {(e['pid'], e['node_id']) for e in events
            if e.get('event') == 'hc_load_done' and e.get('device_tokens', 0) > 0}
    verified = {(e['pid'], e['node_id']): e for e in events
                if e.get('event') == 'hc_kv_verified' and e.get('cache_salt') == salt
                and e.get('success') is True and e.get('algorithm') == 'sha256'
                and e.get('missing_pages') == 0 and e.get('mismatched_pages') == 0}
    good = set()
    for e in events:
        if (e.get('event') != 'match' or e.get('cache_salt') != salt
                or e.get('hit_tokens') != 0 or e.get('host_hit', 0) < minimum):
            continue
        pair = (e['pid'], e.get('host_node'))
        v = verified.get(pair)
        if pair in submitted & done and v and v.get('tokens', 0) >= e['host_hit']:
            good.add(e['pid'])
    return sorted(good) if good == expected else None


def main():
    p = H.parser(__doc__)
    p.add_argument('--pressure-len', type=int, default=8192)
    p.add_argument('--max-rounds', type=int, default=16)
    p.add_argument('--expected-ranks', type=int, default=8)
    a = C.setup(p.parse_args(), 't2_hicache_load_back')
    r = C.Report(a, 't2_hicache_load_back')
    if min(a.expected_ranks, a.max_rounds, a.pressure_len) <= 0:
        p.error('ranks, rounds and pressure-len must be positive')
    evidence = {'config': vars(a).copy(), 'rounds': []}
    start = C.mark_events(a)
    request = Path(a.run_dir) / 'kv_verify.request'
    owned = False
    try:
        salt = H.prepare(a)
        evidence['cache_salt'] = salt
        with request.open('x') as f:
            owned = True
            f.write(salt)
        run_test(a, r, salt, start, evidence)
    except Exception as exc:
        r.check(False, 'experiment completed without exception', repr(exc))
    finally:
        if owned:
            request.unlink(missing_ok=True)
        es = C.events_since(a, start)
        evidence['events'] = es
        r.check(not [e for e in es if e.get('event') == 'hc_observer_error'],
                'HiCache observer has no errors')
        Path(a.out + '.artifacts.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    return r.finish()


def run_test(a, r, salt, start, evidence):
    tokens = C.ids(a.prompt_len, seed=811)
    baseline_response = C.gen(a, tokens, salt=salt)
    def baseline_ready(events):
        baselines = {(e['pid'], e['node_id']) for e in events
                     if e.get('event') == 'hc_kv_baseline'
                     and e.get('cache_salt') == salt and e.get('pages', 0) > 0}
        pids = {pid for pid, _ in H.backed_nodes(events, salt) & baselines}
        return pids if len(pids) == a.expected_ranks else None
    ready = H.wait_events(a, start, baseline_ready)
    record_request(r, evidence, 'target_initial', -1, 0, a.prompt_len,
                   baseline_response, first_matches(C.events_since(a, start), salt))
    if not r.check(bool(ready), 'KV baseline and backup completed on all ranks',
                   sorted(ready) if ready else 'Restart updated observer with M1_HICACHE_VERIFY=1'):
        return
    recovered = False
    for i in range(a.max_rounds):
        # Increase pressure between probes because probing itself warms the target.
        for j in range(i + 1):
            pressure_salt = salt + '-pressure-' + str(i) + '-' + str(j)
            mark = C.mark_events(a)
            response = C.gen(a, C.ids(a.pressure_len, seed=900 + i * a.max_rounds + j),
                             salt=pressure_salt)
            def all_first(es):
                first = first_matches(es, pressure_salt)
                return first if set(first) == ready else None
            H.wait_events(a, mark, all_first)
            pressure_events = C.events_since(a, mark)
            first = first_matches(pressure_events, pressure_salt)
            record_request(r, evidence, 'pressure', i, j, a.pressure_len, response, first)
            evidence.setdefault('pressure', []).append(dict(
                round=i, request_index=j, cache_salt=pressure_salt, events=pressure_events))
            if not r.check(pressure_is_cold(first, ready),
                           'pressure round %d request %d: first GPU/Host match cold on all ranks' % (i + 1, j + 1),
                           list(first.values())):
                return
        probe_start = C.mark_events(a)
        response, events, matches = H.probe(a, tokens, salt)
        recovered = bool(H.wait_events(a, probe_start, lambda es: verified_restore(
            es, salt, ready, C.aligned(a.prompt_len - 1, a.page))))
        events = C.events_since(a, probe_start)
        record_request(r, evidence, 'target_probe', i, 0, a.prompt_len, response,
                       first_matches(events, salt))
        evidence['rounds'].append({'round': i + 1, 'events': events})
        checks = [e for e in events if e.get('event') == 'hc_kv_verified'
                  and e.get('cache_salt') == salt]
        if any(not e.get('success') for e in checks):
            r.check(False, 'restored KV matches original KV bytes', checks)
            break
        if recovered:
            break
    r.check(recovered, 'all-rank GPU miss -> Host restore completed -> KV fingerprints match',
            'Checks all valid KV pages/layers and DSA index cache; see artifacts for evidence.')


if __name__ == '__main__':
    sys.exit(main())
