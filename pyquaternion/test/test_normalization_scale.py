"""Regression tests for finite nonzero quaternion normalization."""
from __future__ import absolute_import, division

from decimal import Decimal, localcontext
from itertools import permutations
from math import cos, ldexp, pi, sin, sqrt
import sys
import unittest

import numpy as np

from pyquaternion import Quaternion


def decimal_unit(values):
    """Normalize the represented input with independent high precision."""
    with localcontext() as context:
        context.prec = 100
        components = [Decimal.from_float(float(value)) for value in values]
        magnitude = sum(value * value for value in components).sqrt()
        return np.array([float(value / magnitude) for value in components])


def z_rotation(angle):
    return np.array([cos(angle / 2.0), 0.0, 0.0, sin(angle / 2.0)])


class TestScaleSafeNormalization(unittest.TestCase):
    def assert_unit(self, quaternion, expected):
        self.assertTrue(np.all(np.isfinite(quaternion.elements)))
        np.testing.assert_allclose(quaternion.elements, expected,
                                   rtol=3e-15, atol=0.0)
        self.assertAlmostEqual(float(np.dot(quaternion.elements,
                                             quaternion.elements)), 1.0, places=14)

    def test_normalised_and_unit_at_finite_limits(self):
        scales = [sys.float_info.max, sys.float_info.max / 4.0,
                  ldexp(1.0, 600), ldexp(1.0, -600), ldexp(1.0, -1074)]
        for scale in scales:
            values = np.array([scale, -scale, scale, -scale])
            expected = decimal_unit(values)
            for property_name in ('normalised', 'unit'):
                quaternion = Quaternion(values)
                with np.errstate(over='raise', divide='raise', invalid='raise',
                                 under='ignore'):
                    result = getattr(quaternion, property_name)
                self.assertIsNot(result, quaternion)
                self.assert_unit(result, expected)
                np.testing.assert_array_equal(quaternion.elements, values)

    def test_scale_invariance_with_permuted_decimal_oracle(self):
        for components in permutations((1.0, -2.0, 3.0, -4.0)):
            for scale in (1e200, 1e-200, -1e200, -1e-200):
                values = np.array(components) * scale
                expected = decimal_unit(values)
                with np.errstate(over='raise', divide='raise', invalid='raise',
                                 under='ignore'):
                    result = Quaternion(values).normalised
                self.assert_unit(result, expected)

    def test_rotation_matrix_matches_rodrigues_at_extreme_scales(self):
        axis = np.ones(3) / sqrt(3.0)
        skew = np.array([[0.0, -axis[2], axis[1]],
                         [axis[2], 0.0, -axis[0]],
                         [-axis[1], axis[0], 0.0]])
        angle = 2.0 * pi / 3.0
        expected = (cos(angle) * np.eye(3)
                    + (1.0 - cos(angle)) * np.outer(axis, axis)
                    + sin(angle) * skew)
        for scale in (1e200, 1e-200, sys.float_info.max, ldexp(1.0, -1074)):
            quaternion = Quaternion([scale] * 4)
            with np.errstate(over='raise', divide='raise', invalid='raise',
                             under='ignore'):
                matrix = quaternion.rotation_matrix
            np.testing.assert_allclose(matrix, expected, rtol=0.0, atol=8e-16)
            np.testing.assert_allclose(np.dot(matrix.T, matrix), np.eye(3),
                                       rtol=0.0, atol=8e-16)
            self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=14)

    def test_rotate_preserves_container_contract_at_extreme_scales(self):
        # [1,1,1,1] represents the cyclic x->y->z rotation.
        expected = np.array([3.0, 1.0, 2.0])
        vectors = ([1.0, 2.0, 3.0], (1.0, 2.0, 3.0),
                   np.array([1.0, 2.0, 3.0]), Quaternion(vector=[1.0, 2.0, 3.0]))
        for scale in (1e200, 1e-200):
            for vector in vectors:
                quaternion = Quaternion([scale] * 4)
                with np.errstate(over='raise', divide='raise', invalid='raise',
                                 under='ignore'):
                    result = quaternion.rotate(vector)
                self.assertIsInstance(result, type(vector))
                if isinstance(result, Quaternion):
                    self.assertEqual(result.scalar, 0.0)
                    actual = result.vector
                else:
                    actual = result
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-15)

    def test_slerp_follows_closed_form_z_rotation_for_scaled_endpoints(self):
        for scales in ((1e200, 1e-200), (1e-200, 1e200)):
            for sign in (1.0, -1.0):
                for amount in (0.0, 0.25, 0.5, 0.75, 1.0):
                    first = Quaternion(z_rotation(0.4) * scales[0])
                    last = Quaternion(z_rotation(1.2) * scales[1] * sign)
                    with np.errstate(over='raise', divide='raise', invalid='raise',
                                     under='ignore'):
                        result = Quaternion.slerp(first, last, amount)
                    expected = z_rotation(0.4 + 0.8 * amount) * sign
                    self.assert_unit(result, expected)

    def test_intermediates_produce_evenly_spaced_unit_rotations(self):
        for scale in (1e200, 1e-200):
            first = Quaternion([scale, 0.0, 0.0, 0.0])
            last = Quaternion(z_rotation(pi / 2.0) * scale)
            with np.errstate(over='raise', divide='raise', invalid='raise',
                             under='ignore'):
                results = list(Quaternion.intermediates(first, last, 3,
                                                       include_endpoints=True))
            self.assertEqual(len(results), 5)
            for index, result in enumerate(results):
                self.assert_unit(result, z_rotation(index * pi / 8.0))

    def test_integrate_normalizes_before_zero_and_constant_rate_steps(self):
        for scale in (1e200, 1e-200):
            for rate, timestep in ((0.0, 0.25), (0.7, 0.0), (0.7, 0.25)):
                quaternion = Quaternion(z_rotation(0.4) * scale)
                with np.errstate(over='raise', divide='raise', invalid='raise',
                                 under='ignore'):
                    result = quaternion.integrate([0.0, 0.0, rate], timestep)
                self.assertIsNone(result)
                self.assert_unit(quaternion, z_rotation(0.4 + rate * timestep))

    def test_zero_unit_tolerance_and_fast_pade_paths_are_unchanged(self):
        zero = Quaternion(0.0)
        np.testing.assert_array_equal(zero.normalised.elements, np.zeros(4))
        zero._fast_normalise()
        np.testing.assert_array_equal(zero.elements, np.zeros(4))
        near_unit = np.array([1.0 + 2e-15, 0.0, 0.0, 0.0])
        for method in ('_normalise', '_fast_normalise'):
            quaternion = Quaternion(near_unit)
            getattr(quaternion, method)()
            np.testing.assert_array_equal(quaternion.elements, near_unit)
        pade_input = np.array([1.0 + 1e-10, 0.0, 0.0, 0.0])
        pade_expected = pade_input / ((1.0 + np.dot(pade_input, pade_input)) / 2.0)
        quaternion = Quaternion(pade_input)
        quaternion._fast_normalise()
        np.testing.assert_array_equal(quaternion.elements, pade_expected)
        ordinary = np.array([1.0, -2.0, 0.5, 1.5])
        ordinary_expected = ordinary / sqrt(float(np.dot(ordinary, ordinary)))
        for method in ('_normalise', '_fast_normalise'):
            quaternion = Quaternion(ordinary)
            getattr(quaternion, method)()
            np.testing.assert_array_equal(quaternion.elements, ordinary_expected)


if __name__ == '__main__':
    unittest.main()
