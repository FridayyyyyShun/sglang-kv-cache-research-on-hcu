import unittest
from array import array

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import (
    CacheAgnosticPolicy,
    CacheAwarePolicy,
    SchedulePolicy,
)
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.test_utils import CustomTestCase


def _make_req(rid, origin_input_text, origin_input_ids, sampling_params=None, **kwargs):
    if sampling_params is None:
        sampling_params = SamplingParams()
    return Req(
        rid,
        origin_input_text,
        array("q", origin_input_ids),
        sampling_params,
        **kwargs,
    )


class TestSchedulePolicy(CustomTestCase):

    def setUp(self):
        self.tree_cache = RadixCache.create_simulated()

    def test_init_with_cache_aware_policy(self):
        policy = SchedulePolicy(
            policy="lpm",
            tree_cache=self.tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        self.assertEqual(policy.policy, CacheAwarePolicy.LPM)

    def test_init_with_cache_agnostic_policy(self):
        policy = SchedulePolicy(
            policy="fcfs",
            tree_cache=self.tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        self.assertEqual(policy.policy, CacheAgnosticPolicy.FCFS)

    def test_init_with_unknown_policy(self):
        with self.assertRaises(ValueError):
            SchedulePolicy(
                policy="invalid",
                tree_cache=self.tree_cache,
                enable_hierarchical_cache=True,
                enable_priority_scheduling=False,
                schedule_low_priority_values_first=False,
            )

    def test_init_with_disabled_cache(self):
        tree_cache = RadixCache.create_simulated(disable=True)
        policy = SchedulePolicy(
            policy="lpm",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        self.assertEqual(policy.policy, CacheAgnosticPolicy.FCFS)

    def test_calc_priority_fcfs(self):
        tree_cache = RadixCache.create_simulated()
        waiting_queue = [
            _make_req(1, "a b", [1, 2]),
            _make_req(3, "a b c", [1, 2, 3]),
            _make_req(2, "a", [1]),
        ]

        policy = SchedulePolicy(
            policy="fcfs",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue)
        # Check if FCFS keeps the original order
        self.assertEqual(waiting_queue[0].rid, 1)
        self.assertEqual(waiting_queue[1].rid, 3)
        self.assertEqual(waiting_queue[2].rid, 2)

    def test_calc_priority_priority_enabled_fcfs_scheduling(self):
        tree_cache = RadixCache.create_simulated()
        r1 = _make_req(1, "a b", [1, 2])
        r2 = _make_req(3, "a b c", [1, 2, 3])
        r3 = _make_req(2, "a", [1])
        r1.priority, r1.time_stats.wait_queue_entry_time = 1, 1
        r2.priority, r2.time_stats.wait_queue_entry_time = 0, 1
        r3.priority, r3.time_stats.wait_queue_entry_time = 0, 0

        waiting_queue = [r1, r2, r3]

        policy = SchedulePolicy(
            policy="fcfs",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=True,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue)

        # Check if priority enabled fcfs ordering is applied.
        self.assertEqual(waiting_queue[0].rid, 1)
        self.assertEqual(waiting_queue[1].rid, 2)
        self.assertEqual(waiting_queue[2].rid, 3)

    def test_calc_priority_priority_enabled_fcfs_scheduling_with_low_priority_values_first(
        self,
    ):
        tree_cache = RadixCache.create_simulated()
        r1 = _make_req(1, "a b", [1, 2])
        r2 = _make_req(3, "a b c", [1, 2, 3])
        r3 = _make_req(2, "a", [1])
        r1.priority, r1.time_stats.wait_queue_entry_time = -1, 1
        r2.priority, r2.time_stats.wait_queue_entry_time = 0, 1
        r3.priority, r3.time_stats.wait_queue_entry_time = 0, 0

        waiting_queue = [r1, r2, r3]

        policy = SchedulePolicy(
            policy="fcfs",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=True,
            schedule_low_priority_values_first=True,
        )
        policy.calc_priority(waiting_queue)
        # Check if priority enabled fcfs ordering is applied.
        self.assertEqual(waiting_queue[0].rid, 1)
        self.assertEqual(waiting_queue[1].rid, 2)
        self.assertEqual(waiting_queue[2].rid, 3)

    def test_calc_priority_longest_output_first_scheduling(self):
        tree_cache = RadixCache.create_simulated()

        waiting_queue = [
            _make_req(1, "a b", [1, 2], SamplingParams(max_new_tokens=1000)),
            _make_req(3, "a b c", [1, 2, 3], SamplingParams(max_new_tokens=10)),
            _make_req(2, "a", [1], SamplingParams(max_new_tokens=100)),
        ]

        policy = SchedulePolicy(
            policy="lof",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue)
        # Check if priority enabled fcfs ordering is applied.
        self.assertEqual(waiting_queue[0].rid, 1)
        self.assertEqual(waiting_queue[1].rid, 2)
        self.assertEqual(waiting_queue[2].rid, 3)

    def test_calc_priority_priority_enabled_longest_output_first_scheduling(self):
        tree_cache = RadixCache.create_simulated()

        waiting_queue = [
            _make_req(1, "a b", [1, 2], SamplingParams(max_new_tokens=1), priority=1),
            _make_req(
                3, "a b c", [1, 2, 3], SamplingParams(max_new_tokens=10), priority=0
            ),
            _make_req(2, "a", [1], SamplingParams(max_new_tokens=100), priority=0),
        ]

        policy = SchedulePolicy(
            policy="lof",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=True,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue)
        # Check if priority enabled fcfs ordering is applied.
        self.assertEqual(waiting_queue[0].rid, 1)
        self.assertEqual(waiting_queue[1].rid, 2)
        self.assertEqual(waiting_queue[2].rid, 3)

    def test_calc_priority_priority_enabled_longest_output_first_scheduling_with_low_priority_values_first(
        self,
    ):
        tree_cache = RadixCache.create_simulated()

        waiting_queue = [
            _make_req(1, "a b", [1, 2], SamplingParams(max_new_tokens=1), priority=0),
            _make_req(
                3, "a b c", [1, 2, 3], SamplingParams(max_new_tokens=10), priority=1
            ),
            _make_req(2, "a", [1], SamplingParams(max_new_tokens=100), priority=1),
        ]

        policy = SchedulePolicy(
            policy="lof",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=True,
            schedule_low_priority_values_first=True,
        )
        policy.calc_priority(waiting_queue)
        # Check if priority enabled fcfs ordering is applied.
        self.assertEqual(waiting_queue[0].rid, 1)
        self.assertEqual(waiting_queue[1].rid, 2)
        self.assertEqual(waiting_queue[2].rid, 3)

    def test_calc_priority_routing_key_scheduling(self):
        """Test routing-key policy: prioritize by routing key frequency in running batch."""
        tree_cache = RadixCache.create_simulated()

        running_reqs = [
            _make_req("r1", "a", [1], routing_key="key_a"),
            _make_req("r2", "b", [2], routing_key="key_a"),
            _make_req("r3", "c", [3], routing_key="key_b"),
        ]
        running_batch = ScheduleBatch(reqs=running_reqs)

        waiting_queue = [
            _make_req("w1", "d", [4], routing_key="key_b"),
            _make_req("w2", "e", [5], routing_key="key_a"),
            _make_req("w3", "f", [6], routing_key="key_c"),
        ]

        policy = SchedulePolicy(
            policy="routing-key",
            tree_cache=tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue, running_batch)

        self.assertEqual(waiting_queue[0].rid, "w2")
        self.assertEqual(waiting_queue[1].rid, "w1")
        self.assertEqual(waiting_queue[2].rid, "w3")

    def test_calc_priority_routing_key_tie_break_by_lexicographic_order(self):
        """Test routing-key policy: tie-break by lexicographic order."""
        tree_cache = RadixCache.create_simulated()

        running_reqs = [
            _make_req("r1", "a", [1], routing_key="key_b"),
            _make_req("r2", "b", [2], routing_key="key_a"),
        ]
        running_batch = ScheduleBatch(reqs=running_reqs)

        waiting_queue = [
            _make_req("w1", "d", [4], routing_key="key_b"),
            _make_req("w2", "e", [5], routing_key="key_a"),
        ]

        policy = SchedulePolicy(
            policy="routing-key",
            tree_cache=tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue, running_batch)

        self.assertEqual(waiting_queue[0].rid, "w2")
        self.assertEqual(waiting_queue[1].rid, "w1")

    def test_calc_priority_routing_key_no_match_deprioritized(self):
        """Test routing-key policy: requests without matching routing keys are deprioritized."""
        tree_cache = RadixCache.create_simulated()

        running_reqs = [
            _make_req("r1", "a", [1], routing_key="key_a"),
            _make_req("r2", "b", [2], routing_key="key_b"),
            _make_req("r3", "c", [3], routing_key="key_c"),
        ]
        running_batch = ScheduleBatch(reqs=running_reqs)

        waiting_queue = [
            _make_req("w1", "d", [4], routing_key="key_d"),
            _make_req("w2", "e", [5], routing_key="key_e"),
            _make_req("w3", "f", [6], routing_key="key_c"),
        ]

        policy = SchedulePolicy(
            policy="routing-key",
            tree_cache=tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue, running_batch)

        self.assertEqual(waiting_queue[0].rid, "w3")
        self.assertEqual(waiting_queue[1].rid, "w1")
        self.assertEqual(waiting_queue[2].rid, "w2")

    def test_calc_priority_routing_key_empty_running_batch(self):
        """Test routing-key policy: empty running batch keeps original order."""
        tree_cache = RadixCache.create_simulated()

        running_batch = ScheduleBatch(reqs=[])

        waiting_queue = [
            _make_req("w1", "d", [4], routing_key="key_a"),
            _make_req("w2", "e", [5], routing_key="key_b"),
            _make_req("w3", "f", [6], routing_key="key_c"),
        ]

        policy = SchedulePolicy(
            policy="routing-key",
            tree_cache=tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue, running_batch)

        self.assertEqual(waiting_queue[0].rid, "w1")
        self.assertEqual(waiting_queue[1].rid, "w2")
        self.assertEqual(waiting_queue[2].rid, "w3")


class TestSchedulePolicyHrrn(CustomTestCase):
    """HRRN: sort by response ratio = waited_tokens / uncached_tokens."""

    def setUp(self):
        self.tree_cache = RadixCache.create_simulated()

    def _make_policy(self):
        return SchedulePolicy(
            policy="hrrn",
            tree_cache=self.tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )

    def test_init_with_hrrn(self):
        self.assertEqual(self._make_policy().policy, CacheAwarePolicy.HRRN)

    def test_calc_priority_hrrn_shortest_first(self):
        """Equal wait, different service time -> shortest uncached first."""
        r_short = _make_req("short", "a", [1])
        r_aged = _make_req("aged", "a" * 3, [1, 2, 3])
        r_long = _make_req("long", "a" * 10, list(range(10)))
        waiting_queue = [r_long, r_aged, r_short]

        # Empty cache -> uncached = 1 / 3 / 10, waited = 1000 for all.
        # Ratios: 1000, 333, 100 -> short, aged, long.
        self._make_policy().calc_priority(waiting_queue, processed_tokens=1000)

        self.assertEqual([r.rid for r in waiting_queue], ["short", "aged", "long"])

    def test_calc_priority_hrrn_aging_overtakes_short(self):
        """A long request that waited long enough must beat a fresh short one."""
        r_long_old = _make_req("long_old", "a" * 100, list(range(100)))
        r_short_new = _make_req("short_new", "a", [1])
        r_long_old.arrival_processed_tokens = 0
        r_short_new.arrival_processed_tokens = 100000
        waiting_queue = [r_short_new, r_long_old]

        # long_old: 100000/100 = 1000; short_new: 0/1 = 0.
        self._make_policy().calc_priority(waiting_queue, processed_tokens=100000)

        self.assertEqual([r.rid for r in waiting_queue], ["long_old", "short_new"])

    def test_calc_priority_hrrn_queue_overflow_falls_back_to_fcfs(self):
        """Above 128 waiting reqs HRRN degrades to FCFS (no prefix matching)."""
        waiting_queue = [_make_req(f"r{i}", "a", [1]) for i in range(129)]

        self._make_policy().calc_priority(waiting_queue, processed_tokens=10)

        self.assertEqual([r.rid for r in waiting_queue], [f"r{i}" for i in range(129)])

    def test_sort_by_hrrn_cached_prefix_and_tiebreak(self):
        """Cached prefix shortens uncached work; equal ratios tie-break by rid."""
        r_more_cached = _make_req("a", "x" * 100, list(range(100)))
        r_less_cached = _make_req("b", "x" * 100, list(range(100)))
        r_tie = _make_req("c", "x" * 100, list(range(100)))
        for r, matched in (
            (r_more_cached, 90),
            (r_less_cached, 0),
            (r_tie, 0),
        ):
            r.num_matched_prefix_tokens = matched
            r.arrival_processed_tokens = 0

        waiting_queue = [r_tie, r_less_cached, r_more_cached]
        # 10 uncached -> 100; 100 uncached -> 10 (b before c).
        SchedulePolicy._sort_by_hrrn(waiting_queue, set(), processed_tokens=1000)

        self.assertEqual([r.rid for r in waiting_queue], ["a", "b", "c"])

    def test_sort_by_hrrn_fully_cached_first_and_deprioritized_last(self):
        """Fully cached drains first, deprioritized goes last; int rids work."""
        ready = _make_req(1, "x" * 4, [1, 2, 3, 4])
        normal = _make_req(2, "x" * 4, [1, 2, 3, 5])
        deprioritized = _make_req(3, "x" * 4, [1, 2, 3, 6])
        ready.num_matched_prefix_tokens = 4
        deprioritized.num_matched_prefix_tokens = 0
        for r in (ready, normal, deprioritized):
            r.arrival_processed_tokens = 0

        waiting_queue = [deprioritized, normal, ready]
        SchedulePolicy._sort_by_hrrn(
            waiting_queue, {deprioritized.rid}, processed_tokens=100
        )

        self.assertEqual([r.rid for r in waiting_queue], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
