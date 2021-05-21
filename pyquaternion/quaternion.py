"""
This file is part of the pyquaternion python module

Author:         Alex Pyattaev
Website:        https://github.com/alex.pyattaev/pyquaternion
Documentation:  http://kieranwynn.github.io/pyquaternion/

Version:         0.1.0
License:         The MIT License (MIT)

Copyright (c) 2015 Kieran Wynn (https://github.com/KieranWynn/pyquaternion)
Copyright (c) 2021 Alex Pyattaev

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

quaternion.py - This file defines the core Quaternion class

"""

from __future__ import annotations
import warnings
from math import sqrt, pi, sin, cos, acos, atan2, exp, log
from math import tau

import numpy as np
from numba import deferred_type

from .numba_opt import jitclass, jit_hardcore, double, numba_available


_EPS: float = np.finfo(float).eps * 4.0

if numba_available:
    _spec_Quaternion = [
        ('q', double[:]),  # array field spec for numba
    ]
else:
    _spec_Quaternion = []

_default_q = np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
_default_v = np.zeros(3, dtype=np.float64)


@jitclass(_spec_Quaternion)
class Quaternion:
    """Class to represent a 4-dimensional complex number or quaternion.

    Quaternion objects can be used generically as 4D numbers,
    or as unit quaternions to represent rotations in 3D space.

    Attributes:
        q: Quaternion 4-vector represented as a Numpy array

    """

    def __init__(self, array: np.ndarray = _default_q):
        """Initialise a new Quaternion object.

        See Object Initialisation docs for complete behaviour:

        https://kieranwynn.github.io/pyquaternion/#object-initialisation

        """
        self.q = np.copy(array)

    @staticmethod
    def from_scalar_and_vector(scalar: double = 0.0, vector: np.ndarray = _default_v):
        return _from_scalar_and_vector(scalar, vector)

    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle: double = 0):
        """Initialise from axis and angle representation

        Create a Quaternion by specifying the 3-vector rotation axis and rotation
        angle (in radians) from which the quaternion's rotation should be created.

        Params:
            axis: a valid numpy 3-vector
            angle: a real valued angle in radians
        """
        return _from_axis_angle(axis, angle)

    @staticmethod
    def from_matrix(matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08):
        """Initialise q vector from matrix representation

        Create q vector by specifying the 3x3 rotation or 4x4 transformation matrix
        (as a numpy array) from which the quaternion's rotation should be created.

        """
        return _from_matrix(matrix, rtol, atol)

    @staticmethod
    def random(randv=None):
        """Generate a random unit quaternion.

        Uniformly distributed across the rotation space
        As per: http://planning.cs.uiuc.edu/node198.html
        """
        if randv is None:
            r1, r2, r3 = np.random.random(3)
        else:
            r1, r2, r3 = randv
        rr1 = sqrt(1.0 - r1)
        rr2 = sqrt(r1)

        t1 = tau * r1
        t2 = tau * r2
        return Quaternion(np.array((cos(t2) * rr2, sin(t1) * rr1, cos(t1) * rr1, sin(t2) * rr2)))

    # Representation
    # def __str__(self):
    #     """An informal, nicely printable string representation of the Quaternion object.
    #     """
    #     return "{:.3f} {:+.3f}i {:+.3f}j {:+.3f}k".format(self.q[0], self.q[1], self.q[2], self.q[3])

    # def __repr__(self):
    #     """The 'official' string representation of the Quaternion object.
    #
    #     This is a string representation of a valid Python expression that could be used
    #     to recreate an object with the same value (given an appropriate environment)
    #     """
    #     return "Quaternion({!r}, {!r}, {!r}, {!r})".format(self.q[0], self.q[1], self.q[2], self.q[3])

    # def __format__(self, formatstr):
    #     """Inserts a customisable, nicely printable string representation of the Quaternion object
    #
    #     The syntax for `format_spec` mirrors that of the built in format specifiers for floating point types.
    #     Check out the official Python [format specification mini-language](https://docs.python.org/3.4/library/string.html#formatspec) for details.
    #     """
    #     if formatstr.strip() == '':  # Defualt behaviour mirrors self.__str__()
    #         formatstr = '+.3f'
    #
    #     string = \
    #         "{:" + formatstr + "} " + \
    #         "{:" + formatstr + "}i " + \
    #         "{:" + formatstr + "}j " + \
    #         "{:" + formatstr + "}k"
    #     return string.format(self.q[0], self.q[1], self.q[2], self.q[3])

    def __bool__(self):
        return not (self == _Q0)

    def __nonzero__(self):
        return not (self == _Q0)

    def __invert__(self):
        return self == _Q0

    # Comparison
    def eq(self, other: Quaternion, a_tol=_EPS):
        """Returns true if the following is true for each element:
        `absolute(a - b) <= atol`
        """
        return np.all(np.abs(self.q - other.q) < a_tol)

    # # Negation
    # def __neg__(self):
    #     return self.__class__(array=-self.q)

    # # Absolute value
    # def __abs__(self):
    #     return self.norm

    def __hash__(self):
        return hash(tuple(self.q))

    # Addition
    def add(self, other: Quaternion):
        return Quaternion(self.q + other.q)

    # Subtraction
    def sub(self, other: Quaternion):
        return Quaternion(self.q - other.q)

    # Multiplication
    def mul(self, other: Quaternion):
        return Quaternion(np.dot(self._q_matrix(), other.q))

    def matmul(self, other: Quaternion):
        return self.q.__matmul__(other.q)

    # Division
    def div(self, other: Quaternion):
        if other == _Q0:
            raise ZeroDivisionError("Quaternion divisor must be non-zero")
        return self * other.inverse

    def pow(self, exponent: float):  # Exponentiation
        # source: https://en.wikipedia.org/wiki/Quaternion#Exponential.2C_logarithm.2C_and_power
        exponent = float(exponent)  # Explicitly reject non-real exponents
        norm = self.norm
        if norm > 0.0:
            try:
                n, theta = self.polar_decomposition
            except ZeroDivisionError:
                # quaternion is a real number (no vector or imaginary part)
                return Quaternion(scalar=self.scalar ** exponent)
            q1 = _from_scalar_and_vector(scalar=(self.norm ** exponent))
            q2 = _from_scalar_and_vector(scalar=cos(exponent * theta), vector=(n * sin(exponent * theta)))
            return q1.mul(q2)
        return Quaternion(self.q)

    @property
    def conjugate(self):
        """Quaternion conjugate, encapsulated in a new instance.

        For a unit quaternion, this is the same as the inverse.

        Returns:
            A new Quaternion object clone with its vector part negated
        """
        return Quaternion(_vector_conjugate(self.q))

    @property
    def inverse(self):
        """Inverse of the quaternion object, encapsulated in a new instance.

        For a unit quaternion, this is the inverse rotation, i.e. when combined with the original rotation, will result in the null rotation.

        Returns:
            A new Quaternion object representing the inverse of this object
        """
        ss = np.dot(self.q, self.q)
        if ss > 0:
            return Quaternion(_vector_conjugate(self.q) / ss)
        else:
            raise ZeroDivisionError("a zero quaternion (0 + 0i + 0j + 0k) cannot be inverted")

    @property
    def norm(self):
        """L2 norm of the quaternion 4-vector.

        This should be 1.0 for a unit quaternion (versor)
        Slow but accurate. If speed is a concern, consider using _fast_normalise() instead

        Returns:
            A scalar real number representing the square root of the sum of the squares of the elements of the quaternion.
        """
        return norm(self.q)

    def _normalise(self):
        """Object is guaranteed to be a unit quaternion after calling this
        operation UNLESS the object is equivalent to Quaternion(0)
        """
        n = norm(self.q)
        if n > 0:
            self.q = self.q / n

    def _fast_normalise(self):
        """Normalise the object to a unit quaternion using a fast approximation method if appropriate.

        Object is guaranteed to be a quaternion of approximately unit length
        after calling this operation UNLESS the object is equivalent to Quaternion(0)
        """
        mag_squared = np.dot(self.q, self.q)
        if mag_squared == 0:
            return
        if abs(1.0 - mag_squared) < 2.107342e-08:
            mag = ((1.0 + mag_squared) / 2.0)  # More efficient. Pade approximation valid if error is small
        else:  # Error is too big, take the performance hit to calculate the square root properly
            mag = sqrt(mag_squared)

        self.q = self.q / mag

    @property
    def normalised(self):
        """Get a unit quaternion (versor) copy of this Quaternion object.

        A unit quaternion has a `norm` of 1.0

        Returns:
            A new Quaternion object clone that is guaranteed to be a unit quaternion
        """
        q = Quaternion(self.q)
        q._normalise()
        return q

    @property
    def polar_unit_vector(self):
        vector_length = norm(self.vector)
        if vector_length <= 0.0:
            raise ZeroDivisionError('Quaternion is pure real and does not have a unique unit vector')
        return self.vector / vector_length

    @property
    def polar_angle(self):
        return acos(self.scalar / self.norm)

    @property
    def polar_decomposition(self):
        """
        Returns the unit vector and angle of a non-scalar quaternion according to the following decomposition

        q =  q.norm() * (e ** (q.polar_unit_vector * q.polar_angle))

        source: https://en.wikipedia.org/wiki/Polar_decomposition#Quaternion_polar_decomposition
        """
        return self.polar_unit_vector, self.polar_angle

    @property
    def unit(self):
        return self.normalised

    def is_unit(self, tolerance=_EPS):
        """Determine whether the quaternion is of unit length to within a specified tolerance value.

        Params:
            tolerance: [optional] maximum absolute value by which the norm can differ from 1.0 for the object to be
            considered a unit quaternion. Defaults to `1e-14`.

        Returns:
            `True` if the Quaternion object is of unit length to within the specified tolerance value.
             `False` otherwise.
        """
        # if _sum_of_squares is 1, norm is 1. This saves a call to sqrt()
        return abs(1.0 - np.dot(self.q, self.q)) < tolerance

    def _q_matrix(self):
        """Matrix representation of quaternion for multiplication purposes.
        """
        return np.array([
            [self.q[0], -self.q[1], -self.q[2], -self.q[3]],
            [self.q[1], self.q[0], -self.q[3], self.q[2]],
            [self.q[2], self.q[3], self.q[0], -self.q[1]],
            [self.q[3], -self.q[2], self.q[1], self.q[0]]])

    def _q_bar_matrix(self):
        """Matrix representation of quaternion for multiplication purposes.
        """
        return np.array([
            [self.q[0], -self.q[1], -self.q[2], -self.q[3]],
            [self.q[1], self.q[0], self.q[3], -self.q[2]],
            [self.q[2], -self.q[3], self.q[0], self.q[1]],
            [self.q[3], self.q[2], -self.q[1], self.q[0]]])

    def _rotate_quaternion(self, q):
        """Rotate a quaternion vector using the stored rotation.

        Params:
            q: The vector to be rotated, in quaternion form (0 + xi + yj + kz)

        Returns:
            A Quaternion object representing the rotated vector in quaternion from (0 + xi + yj + kz)
        """
        self._normalise()
        return self.mul(q).mul(self.conjugate)

    def rotate(self, vector: np.ndarray):
        """Rotate a 3D vector by the rotation stored in the Quaternion object.

        Params:
            vector: A 3-vector specified as any ordered sequence of 3 real numbers corresponding to x, y, and z values.
                Some types that are recognised are: numpy arrays, lists and tuples.
                A 3-vector can also be represented by a Quaternion object who's scalar part is 0 and vector part is the required 3-vector.
                Thus it is possible to call `Quaternion.rotate(q)` with another quaternion object as an input.

        Returns:
            The rotated vector returned as the same type it was specified at input.

        Raises:
            TypeError: if any of the vector elements cannot be converted to a real number.
            ValueError: if `vector` cannot be interpreted as a 3-vector or a Quaternion object.

        """
        q = _from_scalar_and_vector(scalar=0.0, vector=vector)
        return self._rotate_quaternion(q).vector

    @staticmethod
    def exp(q: Quaternion, tolerance=_EPS):
        """Quaternion Exponential.

        Find the exponential of a quaternion amount.

        Params:
             q: the input quaternion/argument as a Quaternion object.
             tolerance: numeric tolerance for null values

        Returns:
             A quaternion amount representing the exp(q). See [Source](https://math.stackexchange.com/questions/1030737/exponential-function-of-quaternion-derivation for more information and mathematical background).

        Note:
             The method can compute the exponential of any quaternion.
        """
        v_norm = norm(q.vector)
        vec = q.vector
        if v_norm > tolerance:
            vec = vec / v_norm
        magnitude = exp(q.scalar)
        return _from_scalar_and_vector(scalar=magnitude * cos(v_norm), vector=magnitude * sin(v_norm) * vec)

    @staticmethod
    def log(q: Quaternion, tolerance=_EPS):
        """Quaternion Logarithm.

        Find the logarithm of a quaternion amount.

        Params:
             q: the input quaternion/argument as a Quaternion object.
             tolerance: numeric tolerance for null values
        Returns:
             A quaternion amount representing log(q) := (log(|q|), v/|v|acos(w/|q|)).

        Note:
            The method computes the logarithm of general quaternions. See [Source](https://math.stackexchange.com/questions/2552/the-logarithm-of-quaternion/2554#2554) for more details.
        """
        v_norm = norm(q.vector)
        q_norm = q.norm

        if q_norm < tolerance:
            # 0 quaternion - undefined
            return Quaternion(scalar=-np.inf, vector=np.NAN * q.vector)
        if v_norm < tolerance:
            # real quaternions - no imaginary part
            return Quaternion(scalar=log(q_norm), vector=np.zeros(3))
        vec = q.vector / v_norm
        return Quaternion(scalar=log(q_norm), vector=acos(q.scalar / q_norm) * vec)

    @staticmethod
    def exp_map(q: Quaternion, eta: Quaternion):
        """Quaternion exponential map.

        Find the exponential map on the Riemannian manifold described
        by the quaternion space.

        Params:
             q: the base point of the exponential map, i.e. a Quaternion object
           eta: the argument of the exponential map, a tangent vector, i.e. a Quaternion object

        Returns:
            A quaternion p such that p is the endpoint of the geodesic starting at q
            in the direction of eta, having the length equal to the magnitude of eta.

        Note:
            The exponential map plays an important role in integrating orientation
            variations (e.g. angular velocities). This is done by projecting
            quaternion tangent vectors onto the quaternion manifold.
        """
        return q * Quaternion.exp(eta)

    @staticmethod
    def sym_exp_map(q: Quaternion, eta: Quaternion):
        """Quaternion symmetrized exponential map.

        Find the symmetrized exponential map on the quaternion Riemannian
        manifold.

        Params:
             q: the base point as a Quaternion object
           eta: the tangent vector argument of the exponential map
                as a Quaternion object

        Returns:
            A quaternion p.

        Note:
            The symmetrized exponential formulation is akin to the exponential
            formulation for symmetric positive definite tensors [Source](http://www.academia.edu/7656761/On_the_Averaging_of_Symmetric_Positive-Definite_Tensors)
        """
        sqrt_q = q.pow(0.5)
        return sqrt_q.mul(Quaternion.exp(eta)).mul(sqrt_q)

    @staticmethod
    def log_map(q, p):
        """Quaternion logarithm map.

        Find the logarithm map on the quaternion Riemannian manifold.

        Params:
             q: the base point at which the logarithm is computed, i.e.
                a Quaternion object
             p: the argument of the quaternion map, a Quaternion object

        Returns:
            A tangent vector having the length and direction given by the
            geodesic joining q and p.
        """
        return Quaternion.log(q.inverse.mul(p))

    @staticmethod
    def sym_log_map(q: Quaternion, p: Quaternion):
        """Quaternion symmetrized logarithm map.

        Find the symmetrized logarithm map on the quaternion Riemannian manifold.

        Params:
             q: the base point at which the logarithm is computed, i.e.
                a Quaternion object
             p: the argument of the quaternion map, a Quaternion object

        Returns:
            A tangent vector corresponding to the symmetrized geodesic curve formulation.

        Note:
            Information on the symmetrized formulations given in [Source](https://www.researchgate.net/publication/267191489_Riemannian_L_p_Averaging_on_Lie_Group_of_Nonzero_Quaternions).
        """
        inv_sqrt_q = q.pow(-0.5)
        return Quaternion.log(inv_sqrt_q.mul(p).mul(inv_sqrt_q))

    @staticmethod
    def absolute_distance(q0, q1):
        """Quaternion absolute distance.

        Find the distance between two quaternions accounting for the sign ambiguity.

        Params:
            q0: the first quaternion
            q1: the second quaternion

        Returns:
           A positive scalar corresponding to the chord of the shortest path/arc that
           connects q0 to q1.

        Note:
           This function does not measure the distance on the hypersphere, but
           it takes into account the fact that q and -q encode the same rotation.
           It is thus a good indicator for rotation similarities.
        """
        q0_minus_q1 = q0 - q1
        q0_plus_q1 = q0 + q1
        d_minus = q0_minus_q1.norm
        d_plus = q0_plus_q1.norm
        if d_minus < d_plus:
            return d_minus
        else:
            return d_plus

    @staticmethod
    def distance(q0, q1):
        """Quaternion intrinsic distance.

        Find the intrinsic geodesic distance between q0 and q1.

        Params:
            q0: the first quaternion
            q1: the second quaternion

        Returns:
           A positive amount corresponding to the length of the geodesic arc
           connecting q0 to q1.

        Note:
           Although the q0^(-1)*q1 != q1^(-1)*q0, the length of the path joining
           them is given by the logarithm of those product quaternions, the norm
           of which is the same.
        """
        q = Quaternion.log_map(q0, q1)
        return q.norm

    @staticmethod
    def sym_distance(q0, q1):
        """Quaternion symmetrized distance.

        Find the intrinsic symmetrized geodesic distance between q0 and q1.

        Params:
            q0: the first quaternion
            q1: the second quaternion

        Returns:
           A positive amount corresponding to the length of the symmetrized
           geodesic curve connecting q0 to q1.

        Note:
           This formulation is more numerically stable when performing
           iterative gradient descent on the Riemannian quaternion manifold.
           However, the distance between q and -q is equal to pi, rendering this
           formulation not useful for measuring rotation similarities when the
           samples are spread over a "solid" angle of more than pi/2 radians
           (the spread refers to quaternions as point samples on the unit hypersphere).
        """
        q = Quaternion.sym_log_map(q0, q1)
        return q.norm

    @staticmethod
    def slerp(q0, q1, amount=0.5):
        """Spherical Linear Interpolation between quaternions.
        Implemented as described in https://en.wikipedia.org/wiki/Slerp

        Find a valid quaternion rotation at a specified distance along the
        minor arc of a great circle passing through any two existing quaternion
        endpoints lying on the unit radius hypersphere.

        This is a class method and is called as a method of the class itself rather than on a particular instance.

        Params:
            q0: first endpoint rotation as a Quaternion object
            q1: second endpoint rotation as a Quaternion object
            amount: interpolation parameter between 0 and 1. This describes the linear placement position of
                the result along the arc between endpoints; 0 being at `q0` and 1 being at `q1`.
                Defaults to the midpoint (0.5).

        Returns:
            A new Quaternion object representing the interpolated rotation. This is guaranteed to be a unit quaternion.

        Note:
            This feature only makes sense when interpolating between unit quaternions (those lying on the unit radius hypersphere).
                Calling this method will implicitly normalise the endpoints to unit quaternions if they are not already unit length.
        """
        # Ensure quaternion inputs are unit quaternions and 0 <= amount <=1
        q0._fast_normalise()
        q1._fast_normalise()
        amount = np.clip(amount, 0, 1)

        dot = np.dot(q0.q, q1.q)

        # If the dot product is negative, slerp won't take the shorter path.
        # Note that v1 and -v1 are equivalent when the negation is applied to all four components.
        # Fix by reversing one quaternion
        if dot < 0.0:
            q0.q = -q0.q
            dot = -dot

        # sin_theta_0 can not be zero
        if dot > 0.9995:
            qr = Quaternion(q0.q + amount * (q1.q - q0.q))
            qr._fast_normalise()
            return qr

        theta_0 = np.arccos(dot)  # Since dot is in range [0, 0.9995], np.arccos() is safe
        sin_theta_0 = np.sin(theta_0)

        theta = theta_0 * amount
        sin_theta = np.sin(theta)

        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        qr = Quaternion((s0 * q0.q) + (s1 * q1.q))
        qr._fast_normalise()
        return qr

    @staticmethod
    def intermediates(q0, q1, n, include_endpoints=False):
        """Generator method to get an iterable sequence of `n` evenly spaced quaternion
        rotations between any two existing quaternion endpoints lying on the unit
        radius hypersphere.

        This is a convenience function that is based on `Quaternion.slerp()` as defined above.

        This is a class method and is called as a method of the class itself rather than on a particular instance.

        Params:
            q_start: initial endpoint rotation as a Quaternion object
            q_end:   final endpoint rotation as a Quaternion object
            n:       number of intermediate quaternion objects to include within the interval
            include_endpoints: [optional] if set to `True`, the sequence of intermediates
                will be 'bookended' by `q_start` and `q_end`, resulting in a sequence length of `n + 2`.
                If set to `False`, endpoints are not included. Defaults to `False`.

        Yields:
            A generator object iterating over a sequence of intermediate quaternion objects.

        Note:
            This feature only makes sense when interpolating between unit quaternions (those lying on the unit radius
            hypersphere).
            Calling this method will implicitly normalise the endpoints to unit quaternions if they are not already
            unit length.
        """
        step_size = 1.0 / (n + 1)
        if include_endpoints:
            steps = [i * step_size for i in range(0, n + 2)]
        else:
            steps = [i * step_size for i in range(1, n + 1)]
        for step in steps:
            yield Quaternion.slerp(q0, q1, step)

    def derivative(self, rate: np.ndarray):
        """Get the instantaneous quaternion derivative representing a quaternion rotating at a 3D rate vector `rate`

        Params:
            rate: numpy 3-array (or array-like) describing rotation rates about the global x, y and z axes respectively.

        Returns:
            A unit quaternion describing the rotation rate
        """
        return _from_scalar_and_vector(0.5).mul(self).mul(_from_scalar_and_vector(0.0, rate))

    def integrate(self, rate: np.ndarray, timestep: double):
        """Advance a time varying quaternion to its value at a time `timestep` in the future.

        The Quaternion object will be modified to its future value.
        It is guaranteed to remain a unit quaternion.

        Params:

        rate: numpy 3-array (or array-like) describing rotation rates about the
            global x, y and z axes respectively.
        timestep: interval over which to integrate into the future.
            Assuming *now* is `T=0`, the integration occurs over the interval
            `T=0` to `T=timestep`. Smaller intervals are more accurate when
            `rate` changes over time.

        Note:
            The solution is closed form given the assumption that `rate` is constant
            over the interval of length `timestep`.
        """
        self._fast_normalise()

        rotation_vector = rate * timestep
        rotation_norm = norm(rotation_vector)
        if rotation_norm > 0:
            axis = rotation_vector / rotation_norm
            angle = rotation_norm
            q2 = _from_axis_angle(axis=axis, angle=angle)
            self.q = (self * q2).q
            self._fast_normalise()

    @property
    def rotation_matrix(self):
        """Get the 3x3 rotation matrix equivalent of the quaternion rotation.

        Returns:
            A 3x3 orthogonal rotation matrix as a 3x3 Numpy array

        Note:
            This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly
             normalise the Quaternion object to a unit quaternion if it is not already one.

        """
        self._normalise()
        product_matrix = np.dot(self._q_matrix(), self._q_bar_matrix().conj().transpose())
        return product_matrix[1:][:, 1:]

    @property
    def yaw_pitch_roll(self):
        """Get the equivalent yaw-pitch-roll angles aka. intrinsic Tait-Bryan angles following the z-y'-x'' convention

        Returns:
            yaw:    rotation angle around the z-axis in radians, in the range `[-pi, pi]`
            pitch:  rotation angle around the y'-axis in radians, in the range `[-pi/2, -pi/2]`
            roll:   rotation angle around the x''-axis in radians, in the range `[-pi, pi]`

        The resulting rotation_matrix would be R = R_x(roll) R_y(pitch) R_z(yaw)

        Note:
            This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.
        """

        self._normalise()
        yaw = np.arctan2(2 * (self.q[0] * self.q[3] - self.q[1] * self.q[2]),
                         1 - 2 * (self.q[2] ** 2 + self.q[3] ** 2))
        pitch = np.arcsin(2 * (self.q[0] * self.q[2] + self.q[3] * self.q[1]))
        roll = np.arctan2(2 * (self.q[0] * self.q[1] - self.q[2] * self.q[3]),
                          1 - 2 * (self.q[1] ** 2 + self.q[2] ** 2))

        return yaw, pitch, roll

    def get_axis(self, undefined=np.zeros(3)):
        """Get the axis or vector about which the quaternion rotation occurs

        For a null rotation (a purely real quaternion), the rotation angle will
        always be `0`, but the rotation axis is undefined.
        It is by default assumed to be `[0, 0, 0]`.

        Params:
            undefined: [optional] specify the axis vector that should define a null rotation.
                This is geometrically meaningless, and could be any of an infinite set of vectors,
                but can be specified if the default (`[0, 0, 0]`) causes undesired behaviour.

        Returns:
            A Numpy unit 3-vector describing the Quaternion object's axis of rotation.

        Note:
            This feature only makes sense when referring to a unit quaternion.
            Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.
        """
        tolerance = 1e-17
        self._normalise()
        n = norm(self.vector)
        if n < tolerance:
            # Here there are an infinite set of possible axes, use what has been specified as an undefined axis.
            return undefined
        else:
            return self.vector / n

    @property
    def axis(self):
        return self.get_axis()

    @property
    def angle(self):
        """Get the angle (in radians) describing the magnitude of the quaternion rotation about its rotation axis.

        This is guaranteed to be within the range (-pi:pi) with the direction of
        rotation indicated by the sign.

        When a particular rotation describes a 180 degree rotation about an arbitrary
        axis vector `v`, the conversion to axis / angle representation may jump
        discontinuously between all permutations of `(-pi, pi)` and `(-v, v)`,
        each being geometrically equivalent (see Note in documentation).

        Returns:
            A real number in the range (-pi:pi) describing the angle of rotation
                in radians about a Quaternion object's axis of rotation.

        Note:
            This feature only makes sense when referring to a unit quaternion.
            Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.
        """
        self._normalise()
        n = norm(self.vector)
        return _wrap_angle(2.0 * atan2(n, self.scalar))

    @property
    def degrees(self):
        return np.rad2deg(self.angle)

    @property
    def radians(self):
        return self.angle

    @property
    def scalar(self):
        """ Return the real or scalar component of the quaternion object.

        Returns:
            A real number i.e. float
        """
        return self.q[0]

    @property
    def vector(self):
        """ Return the imaginary or vector component of the quaternion object.

        Returns:
            A numpy 3-array of floats. NOT guaranteed to be a unit vector
        """
        return self.q[1:4]

    @property
    def real(self):
        return self.scalar

    @property
    def imaginary(self):
        return self.vector

    @property
    def w(self):
        return self.q[0]

    @property
    def x(self):
        return self.q[1]

    @property
    def y(self):
        return self.q[2]

    @property
    def z(self):
        return self.q[3]

    def __getitem__(self, index):
        index = int(index)
        return self.q[index]

    def __setitem__(self, index, value):
        index = int(index)
        self.q[index] = float(value)

    def copy(self):
        return Quaternion(self.q)

    # def __deepcopy__(self, memo):
    #     result = Quaternion(np.copy(self.q))
    #     memo[id(self)] = result
    #     return result

    def transformation_matrix(self):
        """Get the 4x4 homogeneous transformation matrix equivalent of the quaternion rotation.

        Returns:
            A 4x4 homogeneous transformation matrix as a 4x4 Numpy array

        Note:
            This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.
        """
        q = np.copy(self.q)
        n = np.dot(q, q)
        if n < _EPS:
            return np.identity(4)
        q *= sqrt(2.0 / n)
        q = np.outer(q, q)
        return np.array([
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
            [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

    @staticmethod
    def to_degrees(angle_rad):
        warnings.warn("Use numpy rad2deg", DeprecationWarning)
        if angle_rad is not None:
            return np.rad2deg(angle_rad)

    @staticmethod
    def to_radians(angle_deg):
        warnings.warn("Use numpy deg2rad", DeprecationWarning)
        if angle_deg is not None:
            return np.deg2rad(angle_deg)

    def swing_twist_decomp(self, axis):
        """Perform a Swing*Twist decomposition of a Quaternion. This splits the
        quaternion in two: one containing the rotation around axis (Twist), the
        other containing the rotation around a vector parallel to axis (Swing).
        Returns two quaternions: Swing, Twist.
        source: https://github.com/CCP-NC/soprano/blob/master/soprano/utils.py
        """

        # Current rotation axis
        ra = self.q[1:]
        # Ensure that axis is normalised
        axis_norm = axis / np.linalg.norm(axis)
        # Projection of ra along the given axis
        p = np.dot(ra, axis_norm) * axis_norm
        # Create Twist
        qin = np.array((self.q[0], p[0], p[1], p[2]))
        twist = Quaternion(qin / norm(qin))
        # And Swing
        swing = self * twist.conjugate
        return swing, twist


@jit_hardcore
def _wrap_angle(theta):
    """Helper method: Wrap any angle to lie between -pi and pi

    Odd multiples of pi are wrapped to +pi (as opposed to -pi)
    """
    return ((-theta + pi) % (tau) - pi) * -1.0


@jit_hardcore
def _decomposition_method(R: np.ndarray) -> np.ndarray:
    """ Method supposedly able to deal with non-orthogonal matrices - NON-FUNCTIONAL!
    Based on this method: http://arc.aiaa.org/doi/abs/10.2514/2.4654
    """
    x, y, z = 0, 1, 2  # indices
    K = np.array([
        [R[x, x] - R[y, y] - R[z, z], R[y, x] + R[x, y], R[z, x] + R[x, z], R[y, z] - R[z, y]],
        [R[y, x] + R[x, y], R[y, y] - R[x, x] - R[z, z], R[z, y] + R[y, z], R[z, x] - R[x, z]],
        [R[z, x] + R[x, z], R[z, y] + R[y, z], R[z, z] - R[x, x] - R[y, y], R[x, y] - R[y, x]],
        [R[y, z] - R[z, y], R[z, x] - R[x, z], R[x, y] - R[y, x], R[x, x] + R[y, y] + R[z, z]]
    ])
    K = K / 3.0

    e_vals, e_vecs = np.linalg.eig(K)
    print('Eigenvalues:', e_vals)
    print('Eigenvectors:', e_vecs)
    max_index = np.argmax(e_vals)
    principal_component = e_vecs[max_index]
    return principal_component


@jit_hardcore
def _trace_method(matrix: np.ndarray) -> np.ndarray:
    """
    This code uses a modification of the algorithm described in:
    https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    which is itself based on the method described here:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

    Altered to work with the column vector convention instead of row vectors
    """
    m = matrix.conj().transpose()  # This method assumes row-vector and postmultiplication of that vector
    if m[2, 2] < 0:
        if m[0, 0] > m[1, 1]:
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = [m[1, 2] - m[2, 1], t, m[0, 1] + m[1, 0], m[2, 0] + m[0, 2]]
        else:
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = [m[2, 0] - m[0, 2], m[0, 1] + m[1, 0], t, m[1, 2] + m[2, 1]]
    else:
        if m[0, 0] < -m[1, 1]:
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = [m[0, 1] - m[1, 0], m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], t]
        else:
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = [t, m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0]]

    q = np.array(q, dtype=double)
    q *= 0.5 / sqrt(t)
    return q


@jit_hardcore
def _vector_conjugate(q):
    z = np.zeros(4, dtype=double)
    z[0] = q[0]
    z[1:4] = -q[1:4]
    return z


@jit_hardcore
def _from_scalar_and_vector(scalar: double = 0.0, vector: np.ndarray = _default_v):
    # Keyword arguments provided
    q = np.zeros(4, dtype=double)
    q[0] = scalar
    q[1:] = vector
    return Quaternion(q)


@jit_hardcore
def _from_axis_angle(axis: np.ndarray, angle: double):
    """Initialise from axis and angle representation

    Create a Quaternion by specifying the 3-vector rotation axis and rotation
    angle (in radians) from which the quaternion's rotation should be created.

    Params:
        axis: a valid numpy 3-vector
        angle: a real valued angle in radians
    """
    q = np.zeros(4, dtype=double)
    mag_sq = np.dot(axis, axis)
    if mag_sq == 0.0:
        raise ZeroDivisionError("Provided rotation axis has no length")
    # Ensure axis is in unit vector form
    if abs(1.0 - mag_sq) > _EPS:
        axis = axis / sqrt(mag_sq)
    theta = angle / 2.0
    q[0] = cos(theta)
    q[1:3] = axis * sin(theta)
    return Quaternion(q)


@jit_hardcore
def _from_matrix(matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08):
    """Initialise q vector from matrix representation

    Create q vector by specifying the 3x3 rotation or 4x4 transformation matrix
    (as a numpy array) from which the quaternion's rotation should be created.

    """

    shape = matrix.shape

    if shape == (3, 3):
        R = matrix
    elif shape == (4, 4):
        R = matrix[:-1][:, :-1]  # Upper left 3x3 sub-matrix
    else:
        raise ValueError("Invalid matrix shape: Input must be a 3x3 or 4x4 numpy array or matrix")

    # Check matrix properties
    if not np.allclose(np.dot(R, R.conj().transpose()), np.eye(3), rtol=rtol, atol=atol):
        raise ValueError("Matrix must be orthogonal, i.e. its transpose should be its inverse")
    if not np.isclose(np.linalg.det(R), 1.0, rtol=rtol, atol=atol):
        raise ValueError("Matrix must be special orthogonal i.e. its determinant must be +1.0")

    return Quaternion(_trace_method(R))

@jit_hardcore
def norm(v: np.ndarray) -> float:
    """
    Return norm of a vector
    :param v:
    :return:
    >>> norm(np.zeros(3,dtype=float))
    0.0
    >>> np.isclose(norm(np.ones(3)), sqrt(3))
    True
    >>> v0 = np.random.random(3)
    >>> n = norm(v0)
    >>> np.allclose(n, np.linalg.norm(v0))
    True
    """
    assert v.ndim == 1
    # assert v.dtype != np.complex128
    return sqrt((v * v).sum())

_Q0 = Quaternion.from_scalar_and_vector(scalar=0.0)
Quaternion_type = deferred_type()
Quaternion_type.define(Quaternion.class_type.instance_type)

Q1 = Quaternion.random()
Q2 = Quaternion.random()
print(Q1.add(Q2))
print((Q1.mul(Q2)).transformation_matrix())
print(Q1[1])

Q3 = Q1.copy()
Q1[1] = 4
print(Q1.str(), Q3.str())
print(Q1.rotate(np.array([1, 2, 3], dtype=float)))


