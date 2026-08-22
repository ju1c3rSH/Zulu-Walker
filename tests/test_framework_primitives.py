"""Unit tests for the framework scheduling primitives (Slot, CmdQueue).

Run:  python -m unittest discover -s tests -v
Zero dependencies beyond the stdlib - safe on Windows and on-device.
"""

import threading
import unittest

from framework.cmd_queue import Cmd, CmdQueue
from framework.slot import Slot


class TestSlot(unittest.TestCase):
    def test_empty_until_first_publish(self):
        s = Slot()
        self.assertEqual(s.gen, 0)
        self.assertIsNone(s.load())
        self.assertIsNone(s.latest())

    def test_load_returns_new_then_none_when_unchanged(self):
        s = Slot()
        gen = s.publish("frame-a")
        st = s.load(seen_gen=0)
        self.assertEqual(st.value, "frame-a")
        self.assertEqual(st.gen, gen)

        self.assertIsNone(s.load(seen_gen=gen))  # nothing newer
        s.publish("frame-b")
        st2 = s.load(seen_gen=gen)
        self.assertEqual(st2.value, "frame-b")
        self.assertGreater(st2.gen, gen)

    def test_latest_overwrites_intermediate(self):
        s = Slot()
        s.publish(1)
        s.publish(2)
        s.publish(3)
        self.assertEqual(s.latest().value, 3)  # 1 and 2 are gone forever
        self.assertIs(s.latest().value, s.latest().value)  # stable ref

    def test_concurrent_publish_never_loses_final_value(self):
        s = Slot()
        N = 2000

        def writer(tag):
            for i in range(N):
                s.publish((tag, i))

        t1 = threading.Thread(target=writer, args=("a",))
        t2 = threading.Thread(target=writer, args=("b",))
        t1.start(); t2.start(); t1.join(); t2.join()

        final = s.latest()
        self.assertIsNotNone(final)
        tag, last_i = final.value
        # The winner's own sequence must be complete - no torn writes.
        self.assertEqual(last_i, N - 1)


class TestCmdQueue(unittest.TestCase):
    def test_fifo_order_and_drain_clears(self):
        q = CmdQueue()
        self.assertTrue(q.put(Cmd("A", {"n": 1})))
        self.assertTrue(q.put(Cmd("B")))
        self.assertEqual(len(q), 2)

        out = q.drain()
        self.assertEqual([c.kind for c in out], ["A", "B"])
        self.assertEqual(out[0].payload, {"n": 1})
        self.assertEqual(len(q), 0)
        self.assertEqual(q.drain(), [])

    def test_default_coalesce_drops_same_kind(self):
        q = CmdQueue()
        self.assertTrue(q.put(Cmd("CALIB", {"attempt": 1})))
        self.assertFalse(q.put(Cmd("CALIB", {"attempt": 2})))
        out = q.drain()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].payload["attempt"], 1)  # first kept

    def test_replace_swaps_pending_payload(self):
        q = CmdQueue()
        q.put(Cmd("GOTO", {"x": 1}))
        self.assertTrue(q.put(Cmd("GOTO", {"x": 2}), replace=True))
        out = q.drain()
        self.assertEqual(out[0].payload["x"], 2)

    def test_different_kinds_do_not_merge(self):
        q = CmdQueue()
        q.put(Cmd("EXIT"))
        q.put(Cmd("CALIB"))
        self.assertEqual([c.kind for c in q.drain()], ["EXIT", "CALIB"])

    def test_maxlen_drops_oldest(self):
        q = CmdQueue(maxlen=2)
        q.put(Cmd("A"))
        q.put(Cmd("B"))
        q.put(Cmd("C"))  # A silently dropped
        self.assertEqual([c.kind for c in q.drain()], ["B", "C"])

    def test_cross_thread_produce_single_consumer(self):
        q = CmdQueue(maxlen=128)  # > producer count: no overflow drops here

        def producer():
            # Unique kinds so nothing is coalesced - this exercises FIFO
            # ordering, not the merge policy.
            for i in range(100):
                q.put(Cmd(f"k{i}", {"i": i}))

        t = threading.Thread(target=producer)
        t.start(); t.join()
        out = q.drain()
        self.assertEqual(len(out), 100)
        self.assertEqual([c.payload["i"] for c in out], list(range(100)))


if __name__ == "__main__":
    unittest.main()
