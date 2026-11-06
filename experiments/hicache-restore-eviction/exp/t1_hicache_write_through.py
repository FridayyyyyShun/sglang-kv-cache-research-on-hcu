# Run from the experiment root (parent of exp/), with local lib/env.sh configured.
# Terminal 1: start the server; wait until ready.
# M1_HICACHE_VERIFY=0 M1_EVICTION_VERIFY=0 bash lib/serve.sh \
#   --hicache on --hc-ratio 4 --hc-policy write_through \
#   --hc-layout page_first --hc-io kernel --policy lru \
#   --radix on --kv bf16 --page 64 --cap 16384 --ctx 32768 --chunk 8192 \
#   --port 31000 --src repo --obs 1 --root /workspace/kv-runtime
# Terminal 2: run the test (replace SERVER_IP with the actual address).
# bash run_hicache.sh --only t1 --host SERVER_IP
# The runner reads .state/last_run; use --run-dir to override.

"""T1: real-model write-through submission and completed Host backup."""
import sys
import common as C
import hicache_common as H


def main():
    a = C.setup(H.parser(__doc__).parse_args(), 't1_hicache_write_through')
    r = C.Report(a, 't1_hicache_write_through')
    salt = H.prepare(a)
    mark = C.mark_events(a)
    result = C.gen(a, C.ids(a.prompt_len, seed=711), salt=salt)
    nodes = H.wait_events(a, mark, lambda events: H.backed_nodes(events, salt))
    r.row(stage='cold', **{k: v for k, v in result.items() if k != 'raw'})
    r.check(result['cached'] == 0, 'unique namespace starts cold', result['cached'])
    r.check(bool(nodes), 'backup submission has a matching completed Host node', nodes)
    r.check(not C.events_since(a, mark, 'hc_observer_error'), 'HiCache observer has no errors')
    return r.finish()


if __name__ == '__main__':
    sys.exit(main())
