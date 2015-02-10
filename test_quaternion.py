# Unit tests for quaternion module

import unittest
import numpy as np
from random import random, uniform
from quaternion import Quaternion
from math import sqrt, pi, sin, cos
 
def randomElements():
    return ( uniform(-1., 1.), uniform(-1., 1.), uniform(-1., 1.), uniform(-1., 1.) )

class TestQuaternionInitialisation(unittest.TestCase):
    
    def test_init_default(self):
        q = Quaternion()
        self.assertIsInstance(q, Quaternion)
        self.assertEqual(q, Quaternion(1., 0., 0., 0.))

    def test_init_copy(self):
        q1 = Quaternion.random()
        q2 = Quaternion(q1)
        self.assertIsInstance(q2, Quaternion)
        self.assertEqual(q2, q1)
        with self.assertRaises(TypeError):
            q3 = Quaternion(None)
        with self.assertRaises(ValueError):
            q4 = Quaternion("String")
    
    def test_init_random(self): #TODO, this *may* fail at random 
        r1 = Quaternion.random()
        r2 = Quaternion.random()
        self.assertAlmostEqual(r1.norm(), 1.0, 14)
        self.assertIsInstance(r1, Quaternion)
        self.assertNotEqual(r1, r2)

    def test_init_from_scalar(self):
        s = random()
        q1 = Quaternion(s)
        q2 = Quaternion(str(s))
        self.assertIsInstance(q1, Quaternion)
        self.assertIsInstance(q2, Quaternion)
        self.assertEqual(q1, Quaternion(s, 0.0, 0.0, 0.0))
        self.assertEqual(q2, Quaternion(s, 0.0, 0.0, 0.0))
        with self.assertRaises(TypeError):
            q = Quaternion(None)
        with self.assertRaises(ValueError):
            q = Quaternion("String")

    def test_init_from_elements(self):
        a, b, c, d = randomElements()
        q1 = Quaternion(a, b, c, d)
        q2 = Quaternion(str(a), str(b), str(c), str(d))
        q3 = Quaternion(a, str(b), c, d)
        self.assertIsInstance(q1, Quaternion)
        self.assertIsInstance(q2, Quaternion)
        self.assertIsInstance(q3, Quaternion)
        self.assertTrue((q1.q == np.array([a, b, c, d])).all())
        self.assertEqual(q1, q2)
        self.assertEqual(q2, q3)
        with self.assertRaises(TypeError):
            q = Quaternion(None, b, c, d)
        with self.assertRaises(ValueError):
            q = Quaternion(a, b, "String", d)
            q = Quaternion(a, b, c)
            q = Quaternion(a, b, c, d, random())
        
    def test_init_from_array(self):
        r = randomElements()
        a = np.array(r)
        q = Quaternion(a)
        self.assertIsInstance(q, Quaternion)
        self.assertEqual(q, Quaternion(*r))
        with self.assertRaises(ValueError):
            q = Quaternion(a[1:4])            # 3-vector
            q = Quaternion(np.hstack((a, a))) # 8-vector
            q = Quaternion(np.array([a, a]))  # 2x4-
        with self.assertRaises(TypeError):
            q = Quaternion(np.array([None, None, None, None]))

    def test_init_from_tuple(self):
        t = randomElements()
        q = Quaternion(t)
        self.assertIsInstance(q, Quaternion)
        self.assertEqual(q, Quaternion(*t))
        with self.assertRaises(ValueError):
            q = Quaternion(t[1:4])  # 3-tuple
            q = Quaternion(t + t)   # 8-tuple
            q = Quaternion((t, t))  # 2x4-tuple
        with self.assertRaises(TypeError):
            q = Quaternion((None, None, None, None))

    def test_init_from_list(self):
        r = randomElements()
        l = list(r)
        q = Quaternion(l)
        self.assertIsInstance(q, Quaternion)
        self.assertEqual(q, Quaternion(*l))
        with self.assertRaises(ValueError):
            q = Quaternion(l[1:4])  # 3-list
            q = Quaternion(l + l)   # 8-list
            q = Quaternion((l, l))  # 2x4-list
        with self.assertRaises(TypeError):
            q = Quaternion([None, None, None, None])

    def test_init_from_explicit_elements(self):
        e1, e2, e3, e4 = randomElements()
        q1 = Quaternion(w=e1, x=e2, y=e3, z=e4)
        q2 = Quaternion(a=e1, b=str(e2), c=e3, d=e4)
        q3 = Quaternion(a=e1, i=e2, j=e3, k=e4)
        q4 = Quaternion(a=e1)
        self.assertIsInstance(q1, Quaternion)
        self.assertIsInstance(q2, Quaternion)
        self.assertIsInstance(q3, Quaternion)
        self.assertIsInstance(q4, Quaternion)
        self.assertEqual(q1, Quaternion(e1, e2, e3, e4))
        self.assertEqual(q1, q2)
        self.assertEqual(q2, q3)
        self.assertEqual(q4, Quaternion(e1))
        with self.assertRaises(TypeError):
            q = Quaternion(a=None, b=e2, c=e3, d=e4)
        with self.assertRaises(ValueError):
            q = Quaternion(a=e1, b=e2, c="String", d=e4)
            q = Quaternion(w=e1, x=e2)
            q = Quaternion(a=e1, b=e2, c=e3, d=e4, e=e1)

    def test_init_from_explicit_component(self):
        a, b, c, d = randomElements()

        # Using 'real' & 'imaginary' notation
        q1 = Quaternion(real=a, imaginary=(b, c, d))
        q2 = Quaternion(real=a, imaginary=[b, c, d])
        q3 = Quaternion(real=a, imaginary=np.array([b, c, d]))
        q4 = Quaternion(real=a)
        q5 = Quaternion(imaginary=np.array([b, c, d]))
        q6 = Quaternion(real=None, imaginary=np.array([b, c, d]))
        self.assertIsInstance(q1, Quaternion)
        self.assertIsInstance(q2, Quaternion)
        self.assertIsInstance(q3, Quaternion)
        self.assertIsInstance(q4, Quaternion)
        self.assertIsInstance(q5, Quaternion)
        self.assertIsInstance(q6, Quaternion)
        self.assertEqual(q1, Quaternion(a, b, c, d))
        self.assertEqual(q1, q2)
        self.assertEqual(q2, q3)
        self.assertEqual(q4, Quaternion(a, 0, 0, 0))
        self.assertEqual(q5, Quaternion(0, b, c, d))
        self.assertEqual(q5, q6)
        with self.assertRaises(ValueError):
            q = Quaternion(real=a, imaginary=[b, c])
            q = Quaternion(real=a, imaginary=(b, c, d, d))

        # Using 'scalar' & 'vector' notation
        q1 = Quaternion(scalar=a, vector=(b, c, d))
        q2 = Quaternion(scalar=a, vector=[b, c, d])
        q3 = Quaternion(scalar=a, vector=np.array([b, c, d]))
        q4 = Quaternion(scalar=a)
        q5 = Quaternion(vector=np.array([b, c, d]))
        q6 = Quaternion(scalar=None, vector=np.array([b, c, d]))
        self.assertIsInstance(q1, Quaternion)
        self.assertIsInstance(q2, Quaternion)
        self.assertIsInstance(q3, Quaternion)
        self.assertIsInstance(q4, Quaternion)
        self.assertIsInstance(q5, Quaternion)
        self.assertIsInstance(q6, Quaternion)
        self.assertEqual(q1, Quaternion(a, b, c, d))
        self.assertEqual(q1, q2)
        self.assertEqual(q2, q3)
        self.assertEqual(q4, Quaternion(a, 0, 0, 0))
        self.assertEqual(q5, Quaternion(0, b, c, d))
        self.assertEqual(q5, q6)
        with self.assertRaises(ValueError):
            q = Quaternion(scalar=a, vector=[b, c])
            q = Quaternion(scalar=a, vector=(b, c, d, d))

    def test_init_from_explicit_rotation_params(self):
        vx = random()
        vy = random()
        vz = random()
        theta = random() * 2.0 * pi

        v1 = (vx, vy, vz) # tuple format
        v2 = [vx, vy, vz] # list format
        v3 = np.array(v2) # array format
        
        q1 = Quaternion(axis=v1, angle=theta)
        q2 = Quaternion(axis=v2, angle=theta)
        q3 = Quaternion(axis=v3, angle=theta)

        # normalise v to a unit vector
        v3 = v3 / np.linalg.norm(v3)

        q4 = Quaternion(angle=theta, axis=v3)

        # Construct the true quaternion
        t = theta / 2.0

        a = cos(t)
        b = v3[0] * sin(t)
        c = v3[1] * sin(t)
        d = v3[2] * sin(t)

        truth = Quaternion(a, b, c, d)

        self.assertEqual(q1, truth)
        self.assertEqual(q2, truth)
        self.assertEqual(q3, truth)
        self.assertEqual(q4, truth)

        with self.assertRaises(ValueError):
            q = Quaternion(angle=theta)
            q = Quaternion(axis=v1)
            q = Quaternion(axis=[b, c], angle=theta)
            q = Quaternion(axis=(b, c, d, d), angle=theta)
        with self.assertRaises(ZeroDivisionError):
            q = Quaternion(axis=[0., 0., 0.], angle=theta)

    def init_from_explicit_matrix(self):
        self.fail()

    def test_init_from_explicit_arrray(self):
        r = randomElements()
        a = np.array(r)
        q = Quaternion(array=a)
        self.assertIsInstance(q, Quaternion)
        self.assertEqual(q, Quaternion(*r))
        with self.assertRaises(ValueError):
            q = Quaternion(array=a[1:4])            # 3-vector
            q = Quaternion(array=np.hstack((a, a))) # 8-vector
            q = Quaternion(array=np.array([a, a]))  # 2x4-matrix
        with self.assertRaises(TypeError):
            q = Quaternion(array=np.array([None, None, None, None]))

    def test_equivalent_initialisations(self):
        a, b, c, d = randomElements()
        q = Quaternion(a, b, c, d)
        self.assertEqual(q, Quaternion(q))
        self.assertEqual(q, Quaternion(np.array([a, b, c, d])))
        self.assertEqual(q, Quaternion((a, b, c, d)))
        self.assertEqual(q, Quaternion([a, b, c, d]))
        self.assertEqual(q, Quaternion(w=a, x=b, y=c, z=d))
        self.assertEqual(q, Quaternion(array=np.array([a, b, c, d])))

class TestQuaternionRepresentation(unittest.TestCase):

    def test_repr(self):
        q = Quaternion(2.13, -4.15, 3.2, 9.02)
        self.assertEqual(repr(q), "2.130 -4.150i +3.200j +9.020k")

class TestQuaternionArithmetic(unittest.TestCase): 

    def test_equality(self):
        r = randomElements()
        self.assertEqual(Quaternion(*r), Quaternion(*r))
        q = Quaternion(*r)
        self.assertEqual(q, q)
        # Equality should work with other types, if they can be interpreted as quaternions
        self.assertEqual(q, r)
        self.assertEqual(Quaternion(1., 0., 0., 0.), 1.0)
        self.assertEqual(Quaternion(1., 0., 0., 0.), "1.0")
        self.assertNotEqual(q, q + Quaternion(0.0, 0.002, 0.0, 0.0))

        # Equality should also cover small rounding and floating point errors
        self.assertEqual(Quaternion(1., 0., 0., 0.), Quaternion(1.0 - 1e-14, 0., 0., 0.))
        self.assertNotEqual(Quaternion(1., 0., 0., 0.), Quaternion(1.0 - 1e-13, 0., 0., 0.))
        with self.assertRaises(TypeError):
            q == None
        with self.assertRaises(ValueError):
            q == 's'

    def test_assignment(self):
        a, b, c, d = randomElements()
        q1 = Quaternion(a, b, c, d)
        q2 = Quaternion(a, b*0.1, c+0.3, d)
        self.assertNotEqual(q1, q2)
        q2 = q1
        self.assertEqual(q1, q2)

    def test_unary_minus(self):
        a, b, c, d = randomElements()
        q = Quaternion(a, b, c, d)
        self.assertEqual(-q, Quaternion(-a, -b, -c, -d))

    def test_bool(self):
        self.assertTrue(bool(Quaternion()))
        self.assertFalse(bool(Quaternion(scalar=0.0)))

    def test_add(self):
        r1 = randomElements()
        r2 = randomElements()
        r  = random()
        n = None

        q1 = Quaternion(*r1)
        q2 = Quaternion(*r2)
        q3 = Quaternion(array= np.array(r1) + np.array(r2))
        q4 = Quaternion(array= np.array(r2) + np.array([r, 0.0, 0.0, 0.0]))
        self.assertEqual(q1 + q2, q3)
        q1 += q2
        self.assertEqual(q1, q3)
        self.assertEqual(q2 + r, q4)
        self.assertEqual(r + q2, q4)

        with self.assertRaises(TypeError):
            q1 += n
            n += q1

    def test_subtract(self):
        r1 = randomElements()
        r2 = randomElements()
        r  = random()
        n = None

        q1 = Quaternion(*r1)
        q2 = Quaternion(*r2)
        q3 = Quaternion(array= np.array(r1) - np.array(r2))
        q4 = Quaternion(array= np.array(r2) - np.array([r, 0.0, 0.0, 0.0]))
        self.assertEqual(q1 - q2, q3)
        q1 -= q2
        self.assertEqual(q1, q3)
        self.assertEqual(q2 - r, q4)
        self.assertEqual(r - q2, -q4)

        with self.assertRaises(TypeError):
            q1 -= n
            n -= q1

    def test_multiplication_of_bases(self):
        one = Quaternion(1.0, 0.0, 0.0, 0.0)
        i   = Quaternion(0.0, 1.0, 0.0, 0.0)
        j   = Quaternion(0.0, 0.0, 1.0, 0.0)
        k   = Quaternion(0.0, 0.0, 0.0, 1.0)

        self.assertEqual(i * i, j * j)
        self.assertEqual(j * j, k * k)
        self.assertEqual(k * k, i * j * k)
        self.assertEqual(i * j * k, -one)

        self.assertEqual(i * j, k)
        self.assertEqual(i * i, -one)
        self.assertEqual(i * k, -j)
        self.assertEqual(j * i, -k)
        self.assertEqual(j * j, -one)
        self.assertEqual(j * k, i)
        self.assertEqual(k * i, j)
        self.assertEqual(k * j, -i)
        self.assertEqual(k * k, -one)
        self.assertEqual(i * j * k, -one)

    def test_multiply_by_scalar(self):
        a, b, c, d = randomElements()
        q1 = Quaternion(a, b, c, d)
        for s in [30.0, 0.3, -2, -4.7, 0]:
            q2 = Quaternion(s*a, s*b, s*c, s*d)
            q3 = q1
            self.assertEqual(q1 * s, q2) # post-multiply by scalar
            self.assertEqual(s * q1, q2) # pre-multiply by scalar
            q3 *= str(s)
            self.assertEqual(q3, q2)

    def test_multiply_incorrect_type(self):
        q = Quaternion()
        with self.assertRaises(TypeError):
            a = q * None
        with self.assertRaises(ValueError):
            b = q * [1, 1, 1, 1, 1]
            c = q * np.array([[1, 2, 3], [4, 5, 6]])
            d = q * 's'

    def test_divide_by_scalar(self):
        a, b, c, d = randomElements()
        q1 = Quaternion(a, b, c, d)
        for s in [30.0, 0.3, -2, -4.7]:
            q2 = Quaternion(a/s, b/s, c/s, d/s)
            q3 = q1
            self.assertEqual(q1 / s, q2)
            self.assertEqual(s / q1, q2.inverse())
            q3 /= str(s)
            self.assertEqual(q3, q2)

        with self.assertRaises(ZeroDivisionError):
            q4 = q1 / 0.0
        with self.assertRaises(TypeError):
            q4 = q1 / q1
        with self.assertRaises(TypeError):
            q4 = q1 / [a, b, c, d]
        with self.assertRaises(ValueError):
            q4 = q1 / 's'

    def test_squared(self):
        one = Quaternion(1.0, 0.0, 0.0, 0.0)
        i   = Quaternion(0.0, 1.0, 0.0, 0.0)
        j   = Quaternion(0.0, 0.0, 1.0, 0.0)
        k   = Quaternion(0.0, 0.0, 0.0, 1.0)

        self.assertEqual(i**2, j**2)
        self.assertEqual(j**2, k**2) 
        self.assertEqual(k**2, -one)

    def test_power(self):
        q1 = Quaternion.random()
        self.assertEqual(q1 ** 0, Quaternion())
        self.assertEqual(q1 ** 1, q1)
        self.assertEqual(q1 ** 4, q1 * q1 * q1 * q1)

    def test_distributive(self):
        q1 = Quaternion.random()
        q2 = Quaternion.random()
        q3 = Quaternion.random()
        self.assertEqual(q1 * ( q2 + q3 ), q1 * q2 + q1 * q3)

    def test_noncommutative(self):
        q1 = Quaternion.random()
        q2 = Quaternion.random()
        self.assertNotEqual(q1 * q2, q2 * q1)


class TestQuaternionFeatures(unittest.TestCase):

    def test_conjugate(self):
        a, b, c, d = randomElements()
        q1 = Quaternion(a, b, c, d)
        q2 = Quaternion.random()
        self.assertEqual(q1.conjugate(), Quaternion(a, -b, -c, -d))

        self.assertEqual((q1 * q2).conjugate(), q2.conjugate() * q1.conjugate())
        self.assertEqual((q1 + q1.conjugate()) / 2, Quaternion(scalar=q1.scalar()))
        self.assertEqual((q1 - q1.conjugate()) / 2, Quaternion(vector=q1.vector()))
    
    def test_double_conjugate(self):
        q = Quaternion.random()
        self.assertEqual(q, q.conjugate().conjugate())

    def test_norm(self):
        r = randomElements()
        q1 = Quaternion(*r)
        q2 = Quaternion.random()  
        self.assertEqual(q1.norm(), np.linalg.norm(np.array(r)))
        self.assertEqual(q1.magnitude(), np.linalg.norm(np.array(r)))
        # Multiplicative norm
        self.assertAlmostEqual((q1 * q2).norm(), q1.norm() * q2.norm(), 14)
        # Scaled norm
        for s in [30.0, 0.3, -2, -4.7]:
            self.assertAlmostEqual((q1 * s).norm(), q1.norm() * abs(s), 14)

    def test_inverse(self):
        q1 = Quaternion(randomElements())
        q2 = Quaternion.random()
        self.assertEqual(q1 * q1.inverse(), Quaternion(1.0, 0.0, 0.0, 0.0))
        self.assertEqual(q2 * ~q2, Quaternion(1.0, 0.0, 0.0, 0.0))

    def test_normalisation(self): # normalise to unit quaternion
        r = randomElements()
        q1 = Quaternion(*r) 
        v = q1.versor()
        n = q1.normalised()
        self.assertTrue((v.q == q1.q / q1.norm()).all())
        self.assertTrue((n.q == q1.q / q1.norm()).all())
        self.assertAlmostEqual(v.norm(), 1.0, 14)
        self.assertAlmostEqual(n.norm(), 1.0, 14)
        self.assertTrue((q1.axis_as_array() == v.axis_as_array()).all())
        self.assertTrue((q1.axis_as_array() == n.axis_as_array()).all())
        self.assertEqual(q1.angle(), v.angle())
        self.assertEqual(q1.angle(), n.angle())

    def test_q_matrix(self):
        a, b, c, d = randomElements()
        q = Quaternion(a, b, c, d)
        M = np.array([   
            [a, -b, -c, -d],
            [b,  a, -d,  c],
            [c,  d,  a, -b],
            [d, -c,  b,  a]])
        self.assertTrue((q.q_matrix() == M).all())

    def test_q_bar_matrix(self):
        a, b, c, d = randomElements()
        q = Quaternion(a, b, c, d)
        M = np.array([   
            [a, -b, -c, -d],
            [b,  a,  d, -c],
            [c, -d,  a,  b],
            [d,  c, -b,  a]])
        self.assertTrue((q.q_bar_matrix() == M).all())

    def test_rotate(self):
        x = Quaternion(0.0, 1.0, 0.0, 0.0)
        y = Quaternion(0.0, 0.0, 1.0, 0.0)
        z = Quaternion(0.0, 0.0, 0.0, 1.0)

        q1 = Quaternion(axis=x.vector(), angle=  pi)
        q2 = Quaternion(axis=z.vector(), angle=  pi / 2.0)
        q3 = Quaternion(axis=y.vector(), angle= -pi / 2.0)

        self.assertEqual(q1.rotate(z), -z)
        self.assertEqual(q1.rotate(y), -y)
        self.assertEqual(q1.rotate(x), x)
        self.assertEqual(q2.rotate(-x), -y)
        self.assertEqual(q2.rotate(y), -x)
        self.assertEqual(q2.rotate(z), z)
        self.assertEqual(q3.rotate(-z), x)
        self.assertEqual(q3.rotate(x), z)
        self.assertEqual(q3.rotate(y), y)

        self.assertEqual(q1.rotate(2.3*z), -2.3*z)
        
    def rotate_tuple(self):
        x     = (1.0, 0.0, 0.0)
        neg_x = (-1.0, 0.0, 0.0)
        y     = (0.0, 1.0, 0.0)
        neg_y = (0.0, -1.0, 0.0)
        z     = (0.0, 0.0, 1.0)
        neg_z = (0.0, 0.0, -1.0)

        q1 = Quaternion(axis=x, angle=  pi)
        q2 = Quaternion(axis=z, angle=  pi / 2.0)
        q3 = Quaternion(axis=y, angle= -pi / 2.0)

        self.assertEqual(q1.rotate(z), neg_z)
        self.assertEqual(q1.rotate(y), neg_y)
        self.assertEqual(q1.rotate(x), x)
        self.assertEqual(q2.rotate(neg_x), neg_y)
        self.assertEqual(q2.rotate(y), neg_x)
        self.assertEqual(q2.rotate(z), z)
        self.assertEqual(q3.rotate(neg_z), x)
        self.assertEqual(q3.rotate(x), z)
        self.assertEqual(q3.rotate(y), y)


 # See https://github.com/erossignon/pyquaternion/blob/master/test_quaternion.py
 
if __name__ == '__main__':
    unittest.main()