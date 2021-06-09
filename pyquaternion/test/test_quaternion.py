#!/usr/bin python
# -*- coding: utf-8 -*-
"""
This file is part of the pyquaternion python module

Author:         Kieran Wynn
Website:        https://github.com/KieranWynn/pyquaternion
Documentation:  http://kieranwynn.github.io/pyquaternion/

Version:         1.0.0
License:         The MIT License (MIT)

Copyright (c) 2015 Kieran Wynn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

test_quaternion.py - Unit test for quaternion module

"""

import unittest
from math import pi, sin, cos
from random import random

import numpy as np
import pytest

import pyquaternion
from pyquaternion.numba_opt import TypingError
from pyquaternion.quaternion import z_axis, x_axis, to_str

Quaternion = pyquaternion.Quaternion

ALMOST_EQUAL_TOLERANCE = 13


def randomElements():
    return tuple(np.random.uniform(-1, 1, 4))


def test_init_default():
    q = Quaternion()
    assert isinstance(q, Quaternion)
    assert q.eq(Quaternion(np.array((1., 0., 0., 0.))))


def test_init_junk():
    with pytest.raises(TypingError):
        q = Quaternion("blaaa")
    with pytest.raises(TypingError):
        q = Quaternion(None)


def test_init_copy(self):
    q1 = Quaternion.random()
    q2 = q1.copy()
    self.assertIsInstance(q2, Quaternion)
    self.assertTrue(q2.eq(q1))


def test_init_random(self):
    r1 = Quaternion.random()
    r2 = Quaternion.random()
    self.assertAlmostEqual(r1.norm, 1.0, ALMOST_EQUAL_TOLERANCE)
    self.assertIsInstance(r1, Quaternion)
    self.assertNotEqual(r1, r2)  # TODO, this *may* fail at random


def test_init_from_scalar(self):
    s = random()
    q1 = Quaternion.from_scalar_and_vector(s)
    self.assertIsInstance(q1, Quaternion)
    self.assertEqual(q1, Quaternion(np.array((s, 0.0, 0.0, 0.0))))
    with self.assertRaises(TypeError):
        q = Quaternion.from_scalar_and_vector(None)
    with self.assertRaises(TypeError):
        q = Quaternion.from_scalar_and_vector(13, "String")


def test_init_from_elements():
    a, b, c, d = randomElements()
    q1 = Quaternion(np.array([a, b, c, d], dtype=float))

    # assert np.array_equal(q1.q, [a, b, c, d], dtype=float)
    with pytest.raises(ValueError):
        q = Quaternion(np.zeros(3))

    with pytest.raises(ValueError):
        q = Quaternion.from_scalar_and_vector(None, np.array(b, c, d))


def test_init_from_array(self):
    r = randomElements()
    a = np.array(r)
    q = Quaternion(a)
    self.assertIsInstance(q, Quaternion)
    self.assertTrue(np.allclose(q.q, a))
    with self.assertRaises(ValueError):
        q = Quaternion(a[1:4])  # 3-vector
    with self.assertRaises(ValueError):
        q = Quaternion(np.hstack((a, a)))  # 8-vector
    with self.assertRaises(ValueError):
        q = Quaternion(np.array([a, a]))  # 2x4-


def test_init_from_explicit_rotation_params():
    vx = random()
    vy = random()
    vz = random()
    theta = random() * 2.0 * pi

    v1 = np.array([vx, vy, vz], dtype=float)
    v3 = np.copy(v1)

    q1 = Quaternion.from_axis_angle(axis=v1, angle=theta)

    with pytest.raises(ValueError):
        q1 = Quaternion.from_axis_angle(axis=np.zeros(3), angle=theta)
    # normalise v to a unit vector
    v3 = v3 / np.linalg.norm(v3)

    q4 = Quaternion.from_axis_angle(angle=theta, axis=v3)

    # Construct the true quaternion
    t = theta / 2.0

    a = cos(t)
    b = v3[0] * sin(t)
    c = v3[1] * sin(t)
    d = v3[2] * sin(t)

    truth = Quaternion(np.array([a, b, c, d]))

    assert q1.eq(truth)

    assert q4.eq(truth)

    assert Quaternion.from_axis_angle(np.array([1, 0, 0], dtype=float)).eq(Quaternion())

    # Result should be a versor (Unit Quaternion)
    assert abs(q1.norm - 1.0) < ALMOST_EQUAL_TOLERANCE

    with pytest.raises(Exception):
        q = Quaternion.from_axis_angle(angle=theta)
    with pytest.raises(TypingError):
        q = Quaternion.from_axis_angle(axis=[b, c], angle=theta)
    with pytest.raises(TypingError):
        q = Quaternion.from_axis_angle(axis=np.array([1, 2, 3], dtype=int), angle=theta)
    with pytest.raises(TypingError):
        q = Quaternion.from_axis_angle(axis=[b, c], angle=None)


def test_init_from_explicit_matrix():
    def R_z(theta):
        """
        Generate a rotation matrix describing a rotation of theta degrees about the z-axis
        """
        c = cos(theta)
        s = sin(theta)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]])

    v = np.copy(x_axis)
    for angle in [0, pi / 6, pi / 4, pi / 2, pi, 4 * pi / 3, 3 * pi / 2, 2 * pi]:
        R = R_z(angle)  # rotation matrix describing rotation of 90 about +z
        v_prime_r = np.dot(R, v)

        q1 = Quaternion.from_axis_angle(axis=z_axis, angle=angle)
        v_prime_q1 = q1.rotate(v)
        print("=====" + str(angle))
        # assert np.allclose(v_prime_r, v_prime_q1)

        q2 = Quaternion.from_matrix(matrix=R)

        v_prime_q2 = q2.rotate(v)
        print(v_prime_q1)
        print(v_prime_q2)
        print(v_prime_r)
        # assert np.allclose(v_prime_q2, v_prime_r)

    R = np.matrix(np.eye(3))
    q3 = Quaternion.from_matrix(matrix=R)
    v_prime_q3 = q3.rotate(v)
    assert np.allclose(v, v_prime_q3)
    assert q3.eq(Quaternion())

    R[0, 1] += 3  # introduce error to make matrix non-orthogonal
    with pytest.raises(ValueError):
        q4 = Quaternion.from_matrix(matrix=R)


def test_init_from_explicit_matrix_with_optional_tolerance_arguments():
    """
        The matrix defined in this test is orthogonal was carefully crafted
        such that it's orthogonal to a precision of 1e-06, but not to a precision
        of 1e-08.

        Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
    """
    m = [[0.73297226, -0.16524626, -0.65988294, -0.07654548],
         [0.13108627, 0.98617666, -0.10135052, -0.04878795],
         [0.66750896, -0.01221443, 0.74450167, -0.05474513],
         [0, 0, 0, 1, ]]
    npm = np.matrix(m)

    with pytest.raises(ValueError):
        Quaternion.from_matrix(matrix=npm)

    q1 = Quaternion.from_matrix(matrix=npm, atol=1e-6)


def test_str():
    a, b, c, d = randomElements()
    q = Quaternion(np.array([a, b, c, d], dtype=float))
    string = "{:.3f} {:+.3f}i {:+.3f}j {:+.3f}k".format(a, b, c, d)
    assert string == to_str(q)


def test_equality(self):
    r = randomElements()
    self.assertEqual(Quaternion(*r), Quaternion(*r))
    q = Quaternion(*r)
    self.assertEqual(q, q)
    # Equality should work with other types, if they can be interpreted as quaternions
    self.assertEqual(q, r)
    self.assertEqual(Quaternion(1., 0., 0., 0.), 1.0)
    with self.assertRaises(ValueError):
        Quaternion("1.32")
    self.assertNotEqual(q, q + Quaternion(0.0, 0.002, 0.0, 0.0))

    # Equality should also cover small rounding and floating point errors
    self.assertEqual(Quaternion(1., 0., 0., 0.), Quaternion(1.0 - 1e-14, 0., 0., 0.))
    self.assertNotEqual(Quaternion(1., 0., 0., 0.), Quaternion(1.0 - 1e-12, 0., 0., 0.))
    self.assertNotEqual(Quaternion(160., 0., 0., 0.), Quaternion(160.0 - 1e-10, 0., 0., 0.))
    self.assertNotEqual(Quaternion(1600., 0., 0., 0.), Quaternion(1600.0 - 1e-9, 0., 0., 0.))

    with self.assertRaises(TypeError):
        q == None
    with self.assertRaises(ValueError):
        q == 's'


def test_assignment(self):
    a, b, c, d = randomElements()
    q1 = Quaternion(a, b, c, d)
    q2 = Quaternion(a, b * 0.1, c + 0.3, d)
    self.assertNotEqual(q1, q2)
    q2 = q1
    self.assertEqual(q1, q2)


def test_unary_minus(self):
    a, b, c, d = randomElements()
    q = Quaternion(a, b, c, d)
    self.assertEqual(-q, Quaternion(-a, -b, -c, -d))


def test_add(self):
    r1 = randomElements()
    r2 = randomElements()
    r = random()
    n = None

    q1 = Quaternion(*r1)
    q2 = Quaternion(*r2)
    q3 = Quaternion(array=np.array(r1) + np.array(r2))
    q4 = Quaternion(array=np.array(r2) + np.array([r, 0.0, 0.0, 0.0]))
    self.assertEqual(q1 + q2, q3)
    q1 += q2
    self.assertEqual(q1, q3)
    self.assertEqual(q2 + r, q4)
    self.assertEqual(r + q2, q4)

    with self.assertRaises(TypeError):
        q1 += n
    with self.assertRaises(TypeError):
        n += q1


def test_subtract(self):
    r1 = randomElements()
    r2 = randomElements()
    r = random()
    n = None

    q1 = Quaternion(*r1)
    q2 = Quaternion(*r2)
    q3 = Quaternion(array=np.array(r1) - np.array(r2))
    q4 = Quaternion(array=np.array(r2) - np.array([r, 0.0, 0.0, 0.0]))
    self.assertEqual(q1 - q2, q3)
    q1 -= q2
    self.assertEqual(q1, q3)
    self.assertEqual(q2 - r, q4)
    self.assertEqual(r - q2, -q4)

    with self.assertRaises(TypeError):
        q1 -= n
    with self.assertRaises(TypeError):
        n -= q1


def test_multiplication_of_bases():
    one = Quaternion(np.array([1.0, 0.0, 0.0, 0.0]))
    i = Quaternion(np.array([0.0, 1.0, 0.0, 0.0]))
    j = Quaternion(np.array([0.0, 0.0, 1.0, 0.0]))
    k = Quaternion(np.array([0.0, 0.0, 0.0, 1.0]))

    assert i.mul(i).eq(j.mul(j))
    assert j.mul(j).eq(k.mul(k))

    assert k.mul(k).eq(i.mul(j).mul(k))

    assert i.mul(j.mul(k)).eq(one.neg())

    assert i.mul(j).eq(k)
    assert i.mul(i).eq(one.neg())
    assert i.mul(k).eq(j.neg())
    assert j.mul(i).eq(k.neg())
    assert j.mul(j).eq(one.neg())
    assert j.mul(k).eq(i)
    assert k.mul(i).eq(j)
    assert k.mul(j).eq(i.neg())
    assert k.mul(k).eq(one.neg())
    assert i.mul(j).mul(k).eq(one.neg())

    # self.assertEqual(i * i, j * j)
    # self.assertEqual(j * j, k * k)
    # self.assertEqual(k * k, i * j * k)
    # self.assertEqual(i * j * k, -one)
    #
    # self.assertEqual(i * j, k)
    # self.assertEqual(i * i, -one)
    # self.assertEqual(i * k, -j)
    # self.assertEqual(j * i, -k)
    # self.assertEqual(j * j, -one)
    # self.assertEqual(j * k, i)
    # self.assertEqual(k * i, j)
    # self.assertEqual(k * j, -i)
    # self.assertEqual(k * k, -one)
    # self.assertEqual(i * j * k, -one)


def test_multiply_by_scalar():
    a, b, c, d = randomElements()
    q1 = Quaternion(np.array((a, b, c, d)))
    for s in [30.0, 0.3, -2, -4.7, 0]:

        q2 = Quaternion(s * a, s * b, s * c, s * d)
        S = Quaternion.from_scalar_and_vector(s)
        q3 = q1
        assert q1.mul(S).eq(q2)  # post-multiply by scalar
        assert S.mul(q1).eq(q2)  # pre-multiply by scalar
        q3 = q3.mul(S)
        assert q3.eq(q2)


def test_divide():
    r = np.random.rand(4)
    q = Quaternion(r)
    if q:
        assert q.div(q).eq(Quaternion())
    else:
        with pytest.raises(ZeroDivisionError):
            q.div(q)

    with pytest.raises(ZeroDivisionError):
        q.div(Quaternion.from_scalar_and_vector(0.0))


def test_division_of_bases():
    one = Quaternion(np.array([1.0, 0.0, 0.0, 0.0]))
    i = Quaternion(np.array([0.0, 1.0, 0.0, 0.0]))
    j = Quaternion(np.array([0.0, 0.0, 1.0, 0.0]))
    k = Quaternion(np.array([0.0, 0.0, 0.0, 1.0]))

    assert i.div(i).eq(j.div(j))
    # self.assertEqual(j / j, k / k)
    # self.assertEqual(k / k, one)
    # self.assertEqual(k / -k, -one)
    #
    # self.assertEqual(i / j, -k)
    # self.assertEqual(i / i, one)
    # self.assertEqual(i / k, j)
    # self.assertEqual(j / i, k)
    # self.assertEqual(j / j, one)
    # self.assertEqual(j / k, -i)
    # self.assertEqual(k / i, -j)
    # self.assertEqual(k / j, i)
    # self.assertEqual(k / k, one)
    # self.assertEqual(i / -j, k)


def test_divide_by_scalar():
    a, b, c, d = randomElements()
    q1 = Quaternion(a, b, c, d)
    for s in [30.0, 0.3, -2, -4.7]:
        q2 = Quaternion(a / s, b / s, c / s, d / s)
        q3 = q1
        self.assertEqual(q1 / s, q2)
        if q1:
            self.assertEqual(s / q1, q2.inverse)
        else:
            with self.assertRaises(ZeroDivisionError):
                s / q1

        q3 /= s
        self.assertEqual(q3, q2)

    with self.assertRaises(ZeroDivisionError):
        q4 = q1 / 0.0
    with self.assertRaises(TypeError):
        q4 = q1 / None
    with self.assertRaises(ValueError):
        q4 = q1 / 's'


def test_squared():
    one = Quaternion(1.0, 0.0, 0.0, 0.0)
    i = Quaternion(0.0, 1.0, 0.0, 0.0)
    j = Quaternion(0.0, 0.0, 1.0, 0.0)
    k = Quaternion(0.0, 0.0, 0.0, 1.0)

    self.assertEqual(i ** 2, j ** 2)
    self.assertEqual(j ** 2, k ** 2)
    self.assertEqual(k ** 2, -one)


def test_power():
    q1 = Quaternion.random()
    q2 = Quaternion(q1)
    self.assertEqual(q1 ** 0, Quaternion())
    self.assertEqual(q1 ** 1, q1)
    q2 **= 4
    self.assertEqual(q2, q1 * q1 * q1 * q1)
    self.assertEqual((q1 ** 0.5) * (q1 ** 0.5), q1)
    self.assertEqual(q1 ** -1, q1.inverse)
    self.assertEqual(4 ** Quaternion(2), Quaternion(16))
    with self.assertRaises(TypeError):
        q1 ** None
    with self.assertRaises(ValueError):
        q1 ** 's'
    q3 = Quaternion()
    self.assertEqual(q3 ** 0.5, q3)  # Identity behaves as an identity
    self.assertEqual(q3 ** 5, q3)
    self.assertEqual(q3 ** 3.4, q3)
    q4 = Quaternion(scalar=5)  # real number behaves as any other real number would
    self.assertEqual(q4 ** 4, Quaternion(scalar=5 ** 4))


def test_distributive():
    q1 = Quaternion.random()
    q2 = Quaternion.random()
    q3 = Quaternion.random()
    self.assertEqual(q1 * (q2 + q3), q1 * q2 + q1 * q3)


def test_noncommutative():
    q1 = Quaternion.random()
    q2 = Quaternion.random()
    if not q1 == q2:  # Small chance of this happening with random initialisation
        self.assertNotEqual(q1 * q2, q2 * q1)


class TestQuaternionFeatures(unittest.TestCase):

    def test_conjugate(self):
        a, b, c, d = randomElements()
        q1 = Quaternion(a, b, c, d)
        q2 = Quaternion.random()
        self.assertEqual(q1.conjugate, Quaternion(a, -b, -c, -d))

        self.assertEqual((q1 * q2).conjugate, q2.conjugate * q1.conjugate)
        self.assertEqual((q1 + q1.conjugate) / 2, Quaternion(scalar=q1.scalar))
        self.assertEqual((q1 - q1.conjugate) / 2, Quaternion(vector=q1.vector))

    def test_double_conjugate(self):
        q = Quaternion.random()
        self.assertEqual(q, q.conjugate.conjugate)

    def test_norm(self):
        r = randomElements()
        q1 = Quaternion(*r)
        q2 = Quaternion.random()
        self.assertEqual(q1.norm, np.linalg.norm(np.array(r)))
        self.assertEqual(q1.magnitude, np.linalg.norm(np.array(r)))
        # Multiplicative norm
        self.assertAlmostEqual((q1 * q2).norm, q1.norm * q2.norm, ALMOST_EQUAL_TOLERANCE)
        # Scaled norm
        for s in [30.0, 0.3, -2, -4.7]:
            self.assertAlmostEqual((q1 * s).norm, q1.norm * abs(s), ALMOST_EQUAL_TOLERANCE)

    def test_inverse(self):
        q1 = Quaternion(randomElements())
        q2 = Quaternion.random()
        if q1:
            self.assertEqual(q1 * q1.inverse, Quaternion(1.0, 0.0, 0.0, 0.0))
        else:
            with self.assertRaises(ZeroDivisionError):
                q1 * q1.inverse

        self.assertEqual(q2 * q2.inverse, Quaternion(1.0, 0.0, 0.0, 0.0))

    def test_normalisation(self):  # normalise to unit quaternion
        r = randomElements()
        q1 = Quaternion(*r)
        v = q1.unit
        n = q1.normalised

        if q1 == Quaternion(0):  # small chance with random generation
            return  # a 0 quaternion does not normalise

        # Test normalised objects are unit quaternions
        np.testing.assert_almost_equal(v.q, q1.elements / q1.norm, decimal=ALMOST_EQUAL_TOLERANCE)
        np.testing.assert_almost_equal(n.q, q1.elements / q1.norm, decimal=ALMOST_EQUAL_TOLERANCE)
        self.assertAlmostEqual(v.norm, 1.0, ALMOST_EQUAL_TOLERANCE)
        self.assertAlmostEqual(n.norm, 1.0, ALMOST_EQUAL_TOLERANCE)
        # Test axis and angle remain the same
        np.testing.assert_almost_equal(q1.axis, v.axis, decimal=ALMOST_EQUAL_TOLERANCE)
        np.testing.assert_almost_equal(q1.axis, n.axis, decimal=ALMOST_EQUAL_TOLERANCE)
        self.assertAlmostEqual(q1.angle, v.angle, ALMOST_EQUAL_TOLERANCE)
        self.assertAlmostEqual(q1.angle, n.angle, ALMOST_EQUAL_TOLERANCE)
        # Test special case where q is zero
        q2 = Quaternion(0)
        self.assertEqual(q2, q2.normalised)

    def test_is_unit(self):
        q1 = Quaternion()
        q2 = Quaternion(1.0, 0, 0, 0.0001)
        self.assertTrue(q1.is_unit())
        self.assertFalse(q2.is_unit())
        self.assertTrue(q2.is_unit(0.001))

    def test_q_matrix(self):
        a, b, c, d = randomElements()
        q = Quaternion(a, b, c, d)
        M = np.array([
            [a, -b, -c, -d],
            [b, a, -d, c],
            [c, d, a, -b],
            [d, -c, b, a]])
        self.assertTrue(np.array_equal(q._q_matrix(), M))

    def test_q_bar_matrix(self):
        a, b, c, d = randomElements()
        q = Quaternion(a, b, c, d)
        M = np.array([
            [a, -b, -c, -d],
            [b, a, d, -c],
            [c, -d, a, b],
            [d, c, -b, a]])
        self.assertTrue(np.array_equal(q._q_bar_matrix(), M))

    def test_output_of_components(self):
        a, b, c, d = randomElements()
        q = Quaternion(a, b, c, d)
        # Test scalar
        self.assertEqual(q.scalar, a)
        self.assertEqual(q.real, a)
        # Test vector
        self.assertTrue(np.array_equal(q.vector, [b, c, d]))
        self.assertTrue(np.array_equal(q.imaginary, [b, c, d]))
        self.assertEqual(tuple(q.vector), (b, c, d))
        self.assertEqual(list(q.imaginary), [b, c, d])
        self.assertEqual(q.w, a)
        self.assertEqual(q.x, b)
        self.assertEqual(q.y, c)
        self.assertEqual(q.z, d)

    def test_output_of_elements(self):
        r = randomElements()
        q = Quaternion(*r)
        self.assertEqual(tuple(q.elements), r)

    def test_element_access(self):
        r = randomElements()
        q = Quaternion(*r)
        self.assertEqual(q[0], r[0])
        self.assertEqual(q[1], r[1])
        self.assertEqual(q[2], r[2])
        self.assertEqual(q[3], r[3])
        self.assertEqual(q[-1], r[3])
        self.assertEqual(q[-4], r[0])
        with self.assertRaises(TypeError):
            q[None]
        with self.assertRaises(IndexError):
            q[4]
        with self.assertRaises(IndexError):
            q[-5]

    def test_element_assignment(self):
        q = Quaternion()
        self.assertEqual(q[1], 0.0)
        q[1] = 10.0
        self.assertEqual(q[1], 10.0)
        self.assertEqual(q, Quaternion(1.0, 10.0, 0.0, 0.0))
        with self.assertRaises(TypeError):
            q[2] = None
        with self.assertRaises(ValueError):
            q[2] = 's'

    def test_rotate(self):
        q = Quaternion(axis=[1, 1, 1], angle=2 * pi / 3)
        q2 = Quaternion(axis=[1, 0, 0], angle=-pi)
        q3 = Quaternion(axis=[1, 0, 0], angle=pi)
        precision = ALMOST_EQUAL_TOLERANCE
        for r in [1, 3.8976, -69.7, -0.000001]:
            # use np.testing.assert_almost_equal() to compare float sequences
            np.testing.assert_almost_equal(q.rotate((r, 0, 0)), (0, r, 0), decimal=ALMOST_EQUAL_TOLERANCE)
            np.testing.assert_almost_equal(q.rotate([0, r, 0]), [0, 0, r], decimal=ALMOST_EQUAL_TOLERANCE)
            np.testing.assert_almost_equal(q.rotate(np.array([0, 0, r])), np.array([r, 0, 0]),
                                           decimal=ALMOST_EQUAL_TOLERANCE)
            self.assertEqual(q.rotate(Quaternion(vector=[-r, 0, 0])), Quaternion(vector=[0, -r, 0]))
            np.testing.assert_almost_equal(q.rotate([0, -r, 0]), [0, 0, -r], decimal=ALMOST_EQUAL_TOLERANCE)
            self.assertEqual(q.rotate(Quaternion(vector=[0, 0, -r])), Quaternion(vector=[-r, 0, 0]))

            np.testing.assert_almost_equal(q2.rotate((r, 0, 0)), q3.rotate((r, 0, 0)), decimal=ALMOST_EQUAL_TOLERANCE)
            np.testing.assert_almost_equal(q2.rotate((0, r, 0)), q3.rotate((0, r, 0)), decimal=ALMOST_EQUAL_TOLERANCE)
            np.testing.assert_almost_equal(q2.rotate((0, 0, r)), q3.rotate((0, 0, r)), decimal=ALMOST_EQUAL_TOLERANCE)

    def test_conversion_to_matrix(self):
        q = Quaternion.random()
        a, b, c, d = tuple(q.elements)
        R = np.array([
            [a ** 2 + b ** 2 - c ** 2 - d ** 2, 2 * (b * c - a * d), 2 * (a * c + b * d)],
            [2 * (b * c + a * d), a ** 2 - b ** 2 + c ** 2 - d ** 2, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (a * b + c * d), a ** 2 - b ** 2 - c ** 2 + d ** 2]])
        t = np.array([[0], [0], [0]])
        T = np.vstack([np.hstack([R, t]), np.array([0, 0, 0, 1])])
        np.testing.assert_almost_equal(R, q.rotation_matrix, decimal=ALMOST_EQUAL_TOLERANCE)
        np.testing.assert_almost_equal(T, q.transformation_matrix, decimal=ALMOST_EQUAL_TOLERANCE)

        # Test no scaling of rotated vectors
        v1 = np.array([1, 0, 0])
        v2 = np.hstack((np.random.uniform(-10, 10, 3), 1.0))
        v1_ = np.dot(q.rotation_matrix, v1)
        v2_ = np.dot(q.transformation_matrix, v2)
        self.assertAlmostEqual(np.linalg.norm(v1_), 1.0, ALMOST_EQUAL_TOLERANCE)
        self.assertAlmostEqual(np.linalg.norm(v2_), np.linalg.norm(v2), ALMOST_EQUAL_TOLERANCE)

        # Test transformation of vectors is equivalent for quaternion & matrix
        np.testing.assert_almost_equal(v1_, q.rotate(v1), decimal=ALMOST_EQUAL_TOLERANCE)
        np.testing.assert_almost_equal(v2_[0:3], q.rotate(v2[0:3]), decimal=ALMOST_EQUAL_TOLERANCE)

    def test_conversion_to_ypr(self):

        def R_x(theta):
            c = cos(theta)
            s = sin(theta)
            return np.array([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]])

        def R_y(theta):
            c = cos(theta)
            s = sin(theta)
            return np.array([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]])

        def R_z(theta):
            c = cos(theta)
            s = sin(theta)
            return np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]])

        p = np.random.randn(3)
        q = Quaternion.random()
        yaw, pitch, roll = q.yaw_pitch_roll

        p_q = q.rotate(p)
        R_q = q.rotation_matrix

        # build rotation matrix, R = R_z(yaw)*R_y(pitch)*R_x(roll)
        R_ypr = np.dot(R_x(roll), np.dot(R_y(pitch), R_z(yaw)))
        p_ypr = np.dot(R_ypr, p)

        np.testing.assert_almost_equal(p_q, p_ypr, decimal=ALMOST_EQUAL_TOLERANCE)
        np.testing.assert_almost_equal(R_q, R_ypr, decimal=ALMOST_EQUAL_TOLERANCE)

    def test_matrix_io(self):
        v = np.random.uniform(-100, 100, 3)

        for i in range(10):
            q0 = Quaternion.random()
            R = q0.rotation_matrix
            q1 = Quaternion(matrix=R)
            np.testing.assert_almost_equal(q0.rotate(v), np.dot(R, v), decimal=ALMOST_EQUAL_TOLERANCE)
            np.testing.assert_almost_equal(q0.rotate(v), q1.rotate(v), decimal=ALMOST_EQUAL_TOLERANCE)
            np.testing.assert_almost_equal(q1.rotate(v), np.dot(R, v), decimal=ALMOST_EQUAL_TOLERANCE)

            self.assertTrue((q0 == q1) or (q0 == -q1))  # q1 and -q1 are equivalent rotations

    def validate_axis_angle(self, axis, angle):

        def wrap_angle(theta):
            """ Wrap any angle to lie between -pi and pi

            Odd multiples of pi are wrapped to +pi (as opposed to -pi)
            """
            result = ((theta + pi) % (2 * pi)) - pi
            if result == -pi: result = pi
            return result

        theta = wrap_angle(angle)
        v = axis

        q = Quaternion(angle=theta, axis=v)

        v_ = q.axis
        theta_ = q.angle

        if theta == 0.0:  # axis is irrelevant (check defaults to x=y=z)
            np.testing.assert_almost_equal(theta_, 0.0, decimal=ALMOST_EQUAL_TOLERANCE)
            np.testing.assert_almost_equal(v_, np.zeros(3), decimal=ALMOST_EQUAL_TOLERANCE)
            return
        elif abs(theta) == pi:  # rotation in either direction is equivalent
            self.assertTrue(
                np.isclose(theta, pi) or np.isclose(theta, -pi)
                and
                np.isclose(v, v_).all() or np.isclose(v, -v_).all()
            )
        else:
            self.assertTrue(
                np.isclose(theta, theta_) and np.isclose(v, v_).all()
                or
                np.isclose(theta, -theta_) and np.isclose(v, -v_).all()
            )
        # Ensure the returned axis is a unit vector
        np.testing.assert_almost_equal(np.linalg.norm(v_), 1.0, decimal=ALMOST_EQUAL_TOLERANCE)

    def test_conversion_to_axis_angle(self):
        random_axis = np.random.uniform(-1, 1, 3)
        random_axis /= np.linalg.norm(random_axis)

        angles = np.array([-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3]) * pi
        axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), random_axis]

        for v in axes:
            for theta in angles:
                self.validate_axis_angle(v, theta)

    def test_axis_angle_io(self):
        for i in range(20):
            v = np.random.uniform(-1, 1, 3)
            v /= np.linalg.norm(v)
            theta = float(np.random.uniform(-2, 2, 1)) * pi
            self.validate_axis_angle(v, theta)

    def test_exp(self):
        from math import exp
        q = Quaternion(axis=[1, 0, 0], angle=pi)
        exp_q = Quaternion.exp(q)
        self.assertEqual(exp_q, exp(0) * Quaternion(scalar=cos(1.0), vector=[sin(1.0), 0, 0]))

    def test_log(self):
        from math import log
        q = Quaternion(axis=[1, 0, 0], angle=pi)
        log_q = Quaternion.log(q)
        self.assertEqual(log_q, Quaternion(scalar=0, vector=[pi / 2, 0, 0]))

    def test_distance(self):
        q = Quaternion(scalar=0, vector=[1, 0, 0])
        p = Quaternion(scalar=0, vector=[0, 1, 0])
        self.assertEqual(pi / 2, Quaternion.distance(q, p))
        q = Quaternion(angle=pi / 2, axis=[1, 0, 0])
        p = Quaternion(angle=pi / 2, axis=[0, 1, 0])
        self.assertEqual(pi / 3, Quaternion.distance(q, p))
        q = Quaternion(scalar=1, vector=[1, 1, 1])
        p = Quaternion(scalar=-1, vector=[-1, -1, -1])
        p._normalise()
        q._normalise()
        self.assertAlmostEqual(0, Quaternion.distance(q, p), places=8)

    def test_absolute_distance(self):
        q = Quaternion(scalar=0, vector=[1, 0, 0])
        p = Quaternion(scalar=0, vector=[0, 1, 0])
        self.assertEqual((q - p).norm, Quaternion.absolute_distance(q, p))
        q = Quaternion(angle=pi / 2, axis=[1, 0, 0])
        p = Quaternion(angle=pi / 2, axis=[0, 1, 0])
        self.assertEqual((q - p).norm, Quaternion.absolute_distance(q, p))
        q = Quaternion(scalar=0, vector=[1, 0, 0])
        p = Quaternion(scalar=-1, vector=[0, -1, 0])
        self.assertEqual((q + p).norm, Quaternion.absolute_distance(q, p))
        q = Quaternion(scalar=1, vector=[1, 1, 1])
        p = Quaternion(scalar=-1, vector=[-1, -1, -1])
        p._normalise()
        q._normalise()
        self.assertAlmostEqual(0, Quaternion.absolute_distance(q, p), places=8)

    def test_sym_distance(self):
        q = Quaternion(scalar=0, vector=[1, 0, 0])
        p = Quaternion(scalar=0, vector=[0, 1, 0])
        self.assertEqual(pi / 2, Quaternion.sym_distance(q, p))
        q = Quaternion(angle=pi / 2, axis=[1, 0, 0])
        p = Quaternion(angle=pi / 2, axis=[0, 1, 0])
        self.assertAlmostEqual(pi / 3, Quaternion.sym_distance(q, p), places=6)
        q = Quaternion(scalar=0, vector=[1, 0, 0])
        p = Quaternion(scalar=0, vector=[0, -1, 0])
        self.assertEqual(pi / 2, Quaternion.sym_distance(q, p))
        # TODO: this is numerically unstable, previous EPS of 1e-17 was too low for double precision floats
        # q = Quaternion(scalar=1, vector=[1, 1, 1])
        # p = Quaternion(scalar=-1, vector=[-1, -1, -1])
        # p._normalise()
        # q._normalise()
        # self.assertAlmostEqual(pi, Quaternion.sym_distance(q, p), places=8)

    def test_slerp(self):
        q1 = Quaternion(axis=[1, 0, 0], angle=0.0)
        q2 = Quaternion(axis=[1, 0, 0], angle=pi / 2)
        q3 = Quaternion.slerp(q1, q2, 0.5)
        self.assertEqual(q3, Quaternion(axis=[1, 0, 0], angle=pi / 4))

    def test_slerp_extensive(self):
        for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            q1 = Quaternion(axis=axis, angle=0.0)
            q2 = Quaternion(axis=axis, angle=pi / 2.0)
            q3 = Quaternion(axis=axis, angle=pi * 3.0 / 2.0)
            for t in np.arange(0.1, 1, 0.1):
                q4 = Quaternion.slerp(q1, q2, t)
                q5 = Quaternion.slerp(q1, q3, t)
                q6 = Quaternion(axis=axis, angle=t * pi / 2)
                q7 = Quaternion(axis=axis, angle=-t * pi / 2)
                assert q4 == q6 or q4 == -q6
                assert q5 == q7 or q5 == -q7

    def test_interpolate(self):
        q1 = Quaternion(axis=[1, 0, 0], angle=0.0)
        q2 = Quaternion(axis=[1, 0, 0], angle=2 * pi / 3)
        num_intermediates = 3
        base = pi / 6
        list1 = list(Quaternion.intermediates(q1, q2, num_intermediates, include_endpoints=False))
        list2 = list(Quaternion.intermediates(q1, q2, num_intermediates, include_endpoints=True))
        self.assertEqual(len(list1), num_intermediates)
        self.assertEqual(len(list2), num_intermediates + 2)
        self.assertEqual(list1[0], list2[1])
        self.assertEqual(list1[1], list2[2])
        self.assertEqual(list1[2], list2[3])

        self.assertEqual(list2[0], q1)
        self.assertEqual(list2[1], Quaternion(axis=[1, 0, 0], angle=base))
        self.assertEqual(list2[2], Quaternion(axis=[1, 0, 0], angle=2 * base))
        self.assertEqual(list2[3], Quaternion(axis=[1, 0, 0], angle=3 * base))
        self.assertEqual(list2[4], q2)

    def test_differentiation(self):
        q = Quaternion.random()
        omega = np.random.uniform(-1, 1, 3)  # Random angular velocity

        q_dash = 0.5 * q * Quaternion(vector=omega)

        self.assertEqual(q_dash, q.derivative(omega))

    def test_integration(self):
        rotation_rate = [0, 0, 2 * pi]  # one rev per sec around z
        v = [1, 0, 0]  # test vector
        for dt in [0, 0.25, 0.5, 0.75, 1, 2, 10, 1e-10, random() * 10]:  # time step in seconds
            qt = Quaternion()  # no rotation
            qt.integrate(rotation_rate, dt)
            q_truth = Quaternion(axis=[0, 0, 1], angle=dt * 2 * pi)
            a = qt.rotate(v)
            b = q_truth.rotate(v)
            np.testing.assert_almost_equal(a, b, decimal=ALMOST_EQUAL_TOLERANCE)
            self.assertTrue(qt.is_unit())
        # Check integrate() is norm-preserving over many calls
        q = Quaternion()
        for i in range(1000):
            q.integrate([pi, 0, 0], 0.001)
        self.assertTrue(q.is_unit())


class TestQuaternionUtilities(unittest.TestCase):
    def test_copy(self):
        from copy import copy
        q = Quaternion.random()
        q2 = copy(q)
        self.assertEqual(q, q2)
        self.assertFalse(q is q2)
        self.assertTrue(all(q.q == q2.q))

    def test_deep_copy(self):
        from copy import deepcopy
        q = Quaternion.random()
        q2 = deepcopy(q)
        self.assertEqual(q, q2)
        self.assertFalse(q is q2)
        self.assertFalse(q.q is q2.q)


class TestQuaternionHashing(unittest.TestCase):
    def test_equal_quaternions(self):
        q1 = Quaternion(np.array([1, 0, 0, 0]))
        q2 = Quaternion(np.array([1, 0, 0, 0]))

        self.assertEqual(hash(q1), hash(q2))

    def test_unequal_quaternions(self):
        q1 = Quaternion(np.array([1, 0, 0, 0]))
        q2 = Quaternion(np.array([0, 1, 0, 0]))

        self.assertNotEqual(hash(q1), hash(q2))


class TestSwingTwist(unittest.TestCase):
    """
    tests the swing-twist decomposition
    source: https://github.com/CCP-NC/soprano/blob/master/tests/utils_tests.py
    """

    def test_swing_twist(self):
        test_n = 10

        for t_i in range(test_n):
            # Create two quaternions with random rotations
            theta1, theta2 = np.random.random(2) * 2 * np.pi
            ax1 = np.random.random(3)
            ax2 = np.cross(np.random.random(3), ax1)
            ax1 /= np.linalg.norm(ax1)
            ax2 /= np.linalg.norm(ax2)

            q1 = Quaternion(np.array([np.cos(theta1 / 2)] + list(ax1 * np.sin(theta1 / 2))))
            q2 = Quaternion(np.array([np.cos(theta2 / 2)] + list(ax2 * np.sin(theta2 / 2))))

            qT = q1 * q2

            # Now decompose
            qsw, qtw = qT.swing_twist_decomp(ax2)
            # And check
            q1.q *= np.sign(q1.q[0])
            q2.q *= np.sign(q2.q[0])
            qsw.q *= np.sign(qsw.q[0])
            qtw.q *= np.sign(qtw.q[0])

            self.assertTrue(np.allclose(q1.q, qsw.q))
            self.assertTrue(np.allclose(q2.q, qtw.q))


if __name__ == '__main__':
    unittest.main()
