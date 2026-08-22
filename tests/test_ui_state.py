"""Unit tests for framework.ui_state (pure logic, no device deps)."""

import unittest

from framework.ui_state import CooldownGate, HitRegion, bottom_left_region


class TestHitRegion(unittest.TestCase):
    def test_contains_inside_and_edges(self):
        r = HitRegion(x=10, y=20, w=40, h=40)
        self.assertTrue(r.contains(30, 40))          # center
        self.assertTrue(r.contains(10, 60))          # exact corner (bl)
        self.assertTrue(r.contains(50, 20))          # exact corner (tr)

    def test_contains_margin_expands_hitbox(self):
        r = HitRegion(x=10, y=20, w=40, h=40)
        self.assertTrue(r.contains(7, 62, margin=4))     # 3px outside + 4 margin
        self.assertFalse(r.contains(5, 64, margin=4))    # beyond margin

    def test_contains_rejects_outside(self):
        r = HitRegion(x=10, y=20, w=40, h=40)
        self.assertFalse(r.contains(0, 0))
        self.assertFalse(r.contains(100, 100))


class TestBottomLeftRegion(unittest.TestCase):
    def test_matches_legacy_geometry(self):
        # Legacy inline math: by = frame_h - size - 8; bx = margin
        r = bottom_left_region(frame_h=352, size=48, margin=12)
        self.assertEqual((r.x, r.y, r.w, r.h), (12, 352 - 48 - 8, 48, 48))


class TestCooldownGate(unittest.TestCase):
    def test_first_call_allowed_then_blocked_in_window(self):
        g = CooldownGate(cooldown_s=0.3)
        self.assertTrue(g.allow(100.0))
        self.assertFalse(g.allow(100.1))
        self.assertFalse(g.allow(100.29))

    def test_allows_again_after_window(self):
        g = CooldownGate(cooldown_s=0.3)
        self.assertTrue(g.allow(100.0))
        self.assertTrue(g.allow(100.35))

    def test_window_measured_from_last_allow(self):
        g = CooldownGate(cooldown_s=0.3)
        self.assertTrue(g.allow(100.0))
        self.assertTrue(g.allow(100.5))   # allowed, resets window
        self.assertFalse(g.allow(100.7))  # 0.2 after the reset


if __name__ == "__main__":
    unittest.main()
