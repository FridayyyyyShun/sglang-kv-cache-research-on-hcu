"""T1/T2 evidence helpers. Events are correlated by observer PID and node IDs."""
import time
import uuid
import common as C


def parser(desc):
    p = C.base_parser(desc)
    p.add_argument('--prompt-len', type=int, default=4096)
    p.add_argument('--wait-seconds', type=float, default=60)
    return p


def prepare(a):
    if a.prompt_len < 2 * a.page:
        raise ValueError('prompt-len must be >= two pages')
    if not C.wait_ready(a, timeout=30):
        raise RuntimeError('server not ready')
    # Do not use common.flush: it currently swallows HTTP/flush failures.
    # Unique namespace gives each experiment a cold prefix without destructive flush.
    return 'hicache-' + uuid.uuid4().hex


def wait_events(a, since, predicate):
    deadline = time.monotonic() + a.wait_seconds
    while time.monotonic() < deadline:
        events = C.events_since(a, since)
        value = predicate(events)
        if value:
            return value
        time.sleep(.25)
    return None


def backed_nodes(events, salt):
    targets = {(e['pid'], e['node_id']) for e in events
               if e.get('event') == 'insert' and e.get('cache_salt') == salt
               and e.get('node_id') is not None}
    submitted = {(e['pid'], e['node_id']) for e in events
                 if e.get('event') == 'hc_backup_submit' and e.get('success')}
    return {(e['pid'], e['node_id']) for e in events
            if e.get('event') == 'hc_backup_done' and e.get('host_tokens', 0) > 0
            and (e['pid'], e['node_id']) in submitted & targets}


def probe(a, tokens, salt):
    # Return a model-backed match observation without mutating scheduler state.
    mark = C.mark_events(a)
    response = C.gen(a, tokens, salt=salt)
    events = C.events_since(a, mark)
    matches = [e for e in events if e.get('event') == 'match'
               and e.get('cache_salt') == salt]
    return response, events, matches
