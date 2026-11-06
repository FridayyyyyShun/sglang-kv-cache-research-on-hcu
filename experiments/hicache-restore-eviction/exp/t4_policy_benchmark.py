# Run from the experiment root (parent of exp/), with local lib/env.sh configured.
# Terminal 1: start the server; wait until ready.
# M1_HICACHE_VERIFY=0 M1_EVICTION_VERIFY=0 bash lib/serve.sh \
#   --hicache on --hc-ratio 4 --hc-policy write_through \
#   --hc-layout page_first --hc-io kernel --policy lru \
#   --radix on --kv bf16 --page 64 --cap 16384 --ctx 32768 --chunk 8192 \
#   --port 31000 --src repo --obs 1 --root /workspace/kv-runtime
# Terminal 2: run the test (replace SERVER_IP with the actual address).
# bash run_hicache.sh --only t4 --host SERVER_IP --policy lru --cycles 10 --repeats 3
# The runner reads .state/last_run; use --run-dir to override.
# Restart the server for each policy; server and client --policy must agree.
# Flushes cache: dedicated idle server only; keep both verification switches off.

"""T4 workload comparison. Run once per server policy; summarize saved runs.

Unlike t3_eviction_policy.py this measures workloads, not victim ordering.
Each scenario flushes the cache: run only on an idle dedicated test server.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import time
import urllib.request
import uuid

import common as C


def workloads(cycles=1):
    hot = [0, 1] * 6
    cases = {
        'hot_scan': (hot, list(range(10, 18)) + [0, 1] * 4),
        'phase_shift': (hot, [2, 3] * 8),
        'cyclic_over_capacity': (list(range(6)), list(range(6)) * 3),
        'priority_scan': (hot, list(range(10, 18)) + [0, 1] * 4),
        # One old, frequently reused prefix competes with recent one-shot
        # prefixes. Probe it immediately after pressure, before rewarming it.
        'protected_hot_vs_recent_scan': ([0] * 12, [20, 21, 22, 23, 24, 0]),
    }
    if cycles == 1:
        return cases
    # Unique scan keys per cycle; repeated hot keys retain their frequencies.
    scan = [key for cycle in range(cycles)
            for key in list(range(100 + cycle * 8, 108 + cycle * 8)) + [0, 1] * 4]
    protected = [key for cycle in range(cycles)
                 for key in list(range(1000 + cycle * 5, 1005 + cycle * 5)) + [0]]
    return {
        'hot_scan': (hot, scan),
        'phase_shift': (hot, [key for cycle in range(cycles)
                             for key in [2 + cycle * 2, 3 + cycle * 2] * 8]),
        'cyclic_over_capacity': (list(range(6)), list(range(6)) * (3 * cycles)),
        'priority_scan': (hot, scan.copy()),
        'protected_hot_vs_recent_scan': ([0] * 12, protected),
    }


def sample_metrics(samples, length):
    times = sorted(s['latency_ms'] for s in samples)
    return dict(requests=len(samples), latency_mean_ms=statistics.mean(times),
                latency_p50_ms=times[(len(times) - 1) // 2],
                latency_p95_ms=times[max(0, math.ceil(.95 * len(times)) - 1)],
                gpu_hit_fraction=statistics.mean(s['gpu_tokens'] for s in samples) / length,
                host_hit_fraction=statistics.mean(s['host_tokens'] for s in samples) / length)


def summarize(files):
    grouped = {}
    for name in files:
        d = json.loads(Path(name).read_text())
        signature = json.dumps(d['comparison_config'], sort_keys=True)
        for row in d['results']:
            grouped.setdefault((signature, row['scenario'], d['policy']), []).append(row)
    print('| scenario | policy | samples | mean latency ms | GPU hit fraction | Host hit fraction | hot probe GPU fraction |')
    print('|---|---|---:|---:|---:|---:|---:|')
    for (_, scenario, policy), rows in sorted(grouped.items()):
        avg = lambda key: statistics.mean(r[key] for r in rows)
        probes = [r['hot_probe_gpu_fraction'] for r in rows if 'hot_probe_gpu_fraction' in r]
        probe = '%.4f' % statistics.mean(probes) if probes else '-'
        print('| %s | %s | %d | %.2f | %.4f | %.4f | %s |' % (
            scenario, policy, len(rows), avg('latency_mean_ms'),
            avg('gpu_hit_fraction'), avg('host_hit_fraction'), probe))
    if len({k[0] for k in grouped}) > 1:
        print('WARNING: multiple configurations; rows from different configurations are not comparable.')


def main():
    p = C.base_parser(__doc__)
    p.add_argument('--policy', choices=['lru', 'lfu', 'slru', 'priority'])
    p.add_argument('--prompt-len', type=int, default=4096)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--cycles', type=int, default=10,
                   help='measured workload cycles per scenario; 1 reproduces original workload')
    p.add_argument('--scenarios', nargs='+', choices=list(workloads()),
                   help='optional scenario subset, identical across compared policies')
    p.add_argument('--request-timeout', type=float, default=180)
    p.add_argument('--expected-ranks', type=int, default=8)
    p.add_argument('--summarize', nargs='+')
    a = p.parse_args()
    if a.summarize:
        summarize(a.summarize)
        return
    if not a.policy or min(a.repeats, a.expected_ranks, a.cycles, a.request_timeout) <= 0 or a.prompt_len < 2 * a.page:
        p.error('policy required; positive repeats/ranks and prompt-len >= 2 pages required')
    a = C.setup(a, 't4_policy_benchmark_' + a.policy)
    axes = dict(line.split('=', 1) for line in
                (Path(a.run_dir) / 'axes.env').read_text().splitlines() if '=' in line)
    if axes.get('policy') != a.policy:
        p.error('--policy differs from current server axes.env; restart server with matching policy')
    config = {k: axes.get(k) for k in ('host', 'port', 'kv', 'page', 'cap', 'ctx',
              'chunk_prefill', 'hicache', 'hc_ratio', 'hc_policy', 'hc_layout', 'hc_io', 'sglang')}
    config.update(prompt_len=a.prompt_len, page=a.page, cycles=a.cycles,
                  workload_version=2, scenarios=sorted(a.scenarios or workloads()))
    result = dict(policy=a.policy, comparison_config=config, results=[], requests=[])
    path = Path(a.out + '.json')
    try:
        for repeat in range(a.repeats):
            for scenario, (warm, measured) in workloads(a.cycles).items():
                if a.scenarios and scenario not in a.scenarios:
                    continue
                print('[T4] repeat=%d scenario=%s flush; warm=%d measured=%d' %
                      (repeat + 1, scenario, len(warm), len(measured)), flush=True)
                # Strict flush: do not swallow failures as common.flush does.
                req = urllib.request.Request(C._url(a, '/flush_cache?timeout=60'), method='POST')
                with urllib.request.urlopen(req, timeout=90) as response:
                    text = response.read().decode()
                if 'Cache flushed' not in text:
                    raise RuntimeError('Flush not confirmed: ' + text)
                namespace = 't4bench-' + uuid.uuid4().hex
                samples = []
                for index, key in enumerate(warm + measured):
                    mark = C.mark_events(a)
                    begin = time.perf_counter()
                    phase = 'warm' if index < len(warm) else 'measure'
                    print('[T4] %s %s %d/%d key=%d start' %
                          (scenario, phase, index + 1, len(warm) + len(measured), key), flush=True)
                    response = C.http_json(a, '/generate', {
                        'input_ids': C.ids(a.prompt_len, seed=711 + key),
                        'cache_salt': namespace + '-' + str(key),
                        'priority': 100 if scenario == 'priority_scan' and key in (0, 1) else 0,
                        'sampling_params': dict(temperature=0, top_k=1, max_new_tokens=1, ignore_eos=True),
                    }, timeout=a.request_timeout)
                    elapsed = (time.perf_counter() - begin) * 1000
                    print('[T4] done latency_ms=%.2f' % elapsed, flush=True)
                    if index < len(warm):
                        continue
                    # First match per rank avoids counting internal rematches.
                    first = {}
                    for e in C.events_since(a, mark, 'match'):
                        if e.get('cache_salt') == namespace + '-' + str(key):
                            first.setdefault(e['pid'], e)
                    if len(first) != a.expected_ranks:
                        raise RuntimeError('Missing first-match evidence on all ranks')
                    gpu = [e['hit_tokens'] for e in first.values()]
                    host = [e['host_hit'] for e in first.values()]
                    sample = dict(scenario=scenario, repeat=repeat + 1, key=key,
                                  latency_ms=elapsed, gpu_tokens=statistics.mean(gpu),
                                  host_tokens=statistics.mean(host),
                                  cached_tokens=response.get('meta_info', {}).get('cached_tokens'))
                    samples.append(sample)
                    result['requests'].append(sample)
                row = dict(scenario=scenario, repeat=repeat + 1,
                           **sample_metrics(samples, a.prompt_len))
                groups = {}
                for name, selected in [
                    ('gpu_hit', [s for s in samples if s['gpu_tokens'] > 0 and s['host_tokens'] == 0]),
                    ('host_hit', [s for s in samples if s['host_tokens'] > 0]),
                    ('cold', [s for s in samples if s['gpu_tokens'] == 0 and s['host_tokens'] == 0])]:
                    if selected:
                        groups[name] = sample_metrics(selected, a.prompt_len)
                if scenario in ('hot_scan', 'priority_scan', 'protected_hot_vs_recent_scan'):
                    hot_samples = [s for s in samples if s['key'] in (0, 1)]
                    groups['hot_requests'] = sample_metrics(hot_samples, a.prompt_len)
                row['groups'] = groups
                if scenario == 'protected_hot_vs_recent_scan':
                    probe = samples[-1]
                    row.update(hot_probe_latency_ms=probe['latency_ms'],
                               hot_probe_gpu_fraction=probe['gpu_tokens'] / a.prompt_len,
                               hot_probe_host_fraction=probe['host_tokens'] / a.prompt_len)
                    row['hot_probes'] = sample_metrics([s for s in samples if s['key'] == 0], a.prompt_len)
                result['results'].append(row)
                print('[T4 benchmark] ' + json.dumps(row), flush=True)
                path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    finally:
        path.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print('Saved:', path)


if __name__ == '__main__':
    main()
