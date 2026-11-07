import importlib.util
import json
from pathlib import Path
import random
import unittest

spec = importlib.util.spec_from_file_location('geometry', Path(__file__).parents[1] / 'scripts/marshawn_focused_clothing_geometry.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class CameraBalancedStreamTests(unittest.TestCase):
    def setUp(self):
        self.ids = [0] * 6 + [1] * 12 + [2] * 66

    def test_private_draws_and_exact_restart(self):
        random.seed(812)
        before = random.getstate()
        a = m.CameraBalancedStream(self.ids)
        for _ in range(176):
            a.draw()
        state = json.loads(json.dumps(a.state_dict()))
        b = m.CameraBalancedStream(self.ids)
        b.load_state_dict(state)
        for _ in range(16):
            self.assertEqual(a.draw(), b.draw())
        self.assertEqual(a.state_dict(), b.state_dict())
        self.assertEqual(random.getstate(), before)

    def test_all_arm_streams_are_identical(self):
        streams = [m.CameraBalancedStream(self.ids) for _ in m.ARMS]
        for _ in range(10):
            draws = [s.draw() for s in streams]
            self.assertTrue(all(draw == draws[0] for draw in draws))

    def test_reject_changed_membership(self):
        a = m.CameraBalancedStream(self.ids)
        b = m.CameraBalancedStream([1] * 84)
        with self.assertRaises(ValueError):
            b.load_state_dict(a.state_dict())

    def test_camera_balance_not_observation_balance(self):
        a = m.CameraBalancedStream(self.ids)
        counts = [0, 0, 0]
        for _ in range(200):
            for index in a.draw():
                counts[self.ids[index]] += 1
        for count in counts:
            self.assertLess(abs(count / sum(counts) - 1/3), .015)


if __name__ == '__main__':
    unittest.main()
