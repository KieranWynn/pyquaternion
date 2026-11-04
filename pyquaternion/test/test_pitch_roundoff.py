import unittest
import numpy as np
from pyquaternion import Quaternion


class TestPitchRoundoff(unittest.TestCase):
    def test_pitch_at_gimbal_lock_stays_finite(self):
        original = np.array(
            [
                0.5017291299643045,
                -0.4982648695102219,
                0.5017291299603768,
                -0.49826486942832626,
            ]
        )
        for pitch_sign in [1, -1]:
            for quaternion_sign in [1, -1]:
                values = original.copy()
                values[2:] *= pitch_sign
                q = Quaternion(values * quaternion_sign)
                with np.errstate(invalid="raise"):
                    angles = q.yaw_pitch_roll
                self.assertTrue(np.all(np.isfinite(angles)))
                self.assertAlmostEqual(angles[1], pitch_sign * np.pi / 2)

    def test_ordinary_pitch_is_unchanged(self):
        for angle in [-0.8, 0.0, 0.8]:
            q = Quaternion(axis=[0, 1, 0], radians=angle)
            np.testing.assert_allclose(q.yaw_pitch_roll, [0, angle, 0], atol=1e-15)
