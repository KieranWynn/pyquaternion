# Unit tests for quaternion module

import unittest
import numpy as np
from random import random, uniform
from quaternion import Quaternion
 
def randomElements():
    return ( uniform(-1., 1.), uniform(-1., 1.), uniform(-1., 1.), uniform(-1., 1.) )

class TestQuaternionInitialisation(unittest.TestCase):
 
    def setUp(self):
        pass

    # Test initialisation of objects
    def test_init(self):
        a, b, c, d = randomElements()
        q = Quaternion(a, b, c, d)
        self.assertIsInstance(q, Quaternion)

    def test_init_incorrect_type(self):
        self.assertRaises(ValueError, Quaternion, "1 +0i + 0j + 0k")

    def test_init_default(self):
        q = Quaternion()
        self.assertIsInstance(q, Quaternion)
        self.assertEqual(q, Quaternion(1., 0., 0., 0.))

    def test_init_from_array_correct(self):
        q = Quaternion.from_array(np.array([1., 0., 0., 0.]))
        self.assertIsInstance(q, Quaternion)

    def test_init_from_array_incorrect_type(self):
        self.assertRaises(TypeError, Quaternion.from_array, 3.0)

    def test_init_from_array_incorrect_size(self):
        self.assertRaises(RuntimeError, Quaternion.from_array, np.array([1., 0., 0.]))

    def test_init_from_scalar(self):
        s = random()
        self.assertEqual(Quaternion(s), Quaternion(s, 0.0, 0.0, 0.0))

    def test_init_from_tuple(self):
        r = randomElements()
        self.assertEqual(Quaternion(r), Quaternion(*r))

    def test_init_from_list(self):
        r = randomElements()
        self.assertEqual(Quaternion(list(r)), Quaternion(*r))

    def test_init_from_dict(self):
        r = randomElements()
        d = {"a":r[0], "b":r[1], "c":r[2], "d":r[3]}
        self.assertEqual(Quaternion(d), Quaternion(*r))
 
    def test_equivalent_initialisations(self):
        self.assertEqual(Quaternion(1., 0., 0., 0.), Quaternion.from_array(np.array([1., 0., 0., 0.])))
        self.assertEqual(Quaternion(1., 0., 0., 0.), Quaternion.from_elements(1., 0., 0., 0.))

    # def test_init_from_elements_correct(self):
    #     q = Quaternion.from_elements(1., 0., 0., 0.)
    #     self.assertIsInstance(q, Quaternion)

    # def test_init_from_elements_incorrect_type(self):
    #     self.assertRaises(TypeError, Quaternion.from_elements, "1", "0", "0", "0")

    def test_random(self):
        r1 = Quaternion.random()
        (r2, r3, r4) = (Quaternion.random(), Quaternion.random(), Quaternion.random())
        self.assertIsInstance(r1, Quaternion)
        self.assertNotEqual(r1, r3) 
        # There is a chance (albeit small) that random() could generate an equal pair of objects
        # The following significantly reduces likelihood of randomly generating a failing case
        #self.assertNotEqual(r1 + r2, r3 + r4)

    def test_repr(self):
        q = Quaternion(2.13, -4.15, 3.2, 9.02)
        self.assertEqual(repr(q), "2.130 -4.150i +3.200j +9.020k")

class TestQuaternionArithmetic(unittest.TestCase): 

    def test_equality(self):
        r = randomElements()
        self.assertEqual(Quaternion(*r), Quaternion(*r))
        q = Quaternion(*r)
        self.assertEqual(q, q)
        # Equality should cover small rounding and floating point errors
        self.assertEqual(Quaternion(1., 0., 0., 0.), Quaternion(0.99999999, 0., 0., 0.))

    def test_equality_false(self):
        a, b, c, d = randomElements()
        q1 = Quaternion(a, b, c, d)
        q2 = Quaternion(a, b*0.1, c+0.3, d)
        self.assertNotEqual(q1, q2)

    def test_assignment(self):
        a, b, c, d = randomElements()
        q1 = Quaternion(a, b, c, d)
        q2 = Quaternion(a, b*0.1, c+0.3, d)
        self.assertNotEqual(q1, q2)
        q2 = q1
        self.assertEqual(q1, q2)

    def test_add(self):
        r1 = randomElements()
        r2 = randomElements()
        self.assertEqual(
            Quaternion(*r1) + Quaternion(*r2), 
            Quaternion.from_array(np.array(r1) + np.array(r2))
            )

    def test_subtract(self):
        r1 = randomElements()
        r2 = randomElements()
        self.assertEqual(
            Quaternion(*r1) - Quaternion(*r2), 
            Quaternion.from_array(np.array(r1) - np.array(r2))
            )

    def test_multiply_by_quaternion(self):
        """  Requires equality (__eq__) and unary minus (__neg__) to work properly
        """
        one = Quaternion(1.0, 0.0, 0.0, 0.0)
        i   = Quaternion(0.0, 1.0, 0.0, 0.0)
        j   = Quaternion(0.0, 0.0, 1.0, 0.0)
        k   = Quaternion(0.0, 0.0, 0.0, 1.0)

        self.assertNotEqual(i * j, j * i)

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
            self.assertEqual(q1*s, q2) # post-multiply by scalar
            self.assertEqual(s*q1, q2) # pre-multiply by scalar

    def test_multiply(self):
        q = Quaternion.random()
        self.assertEqual(q * 4.0, q * Quaternion(4.0, 0.0, 0.0, 0.0))
        self.assertEqual(q * 0.4, q * Quaternion(0.4, 0.0, 0.0, 0.0))

    def test_multiply_incorrect_type(self):
        self.assertRaises(Exception, Quaternion().__mul__, list(randomElements()))
        self.assertRaises(Exception, Quaternion().__mul__, "Text")
        self.assertRaises(Exception, Quaternion().__mul__, np.array(randomElements()) )


    def test_divide(self):
        pass

    def test_divide_by_scalar(self):
        a, b, c, d = randomElements()
        q1 = Quaternion(a, b, c, d)
        for s in [30.0, 0.3, -2, -4.7]:
            q2 = Quaternion(a/s, b/s, c/s, d/s)
            self.assertEqual(q1/s, q2)
        
        s = 0.0
        with self.assertRaises(ZeroDivisionError):
            q3 = q1/s

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

    def test_unary_minus(self):
        a, b, c, d = randomElements()
        q = Quaternion(a, b, c, d)
        self.assertEqual(-q, Quaternion(-a, -b, -c, -d))

class TestQuaternionFeatures(unittest.TestCase):

    def test_conjugate(self):
        a, b, c, d = randomElements()
        self.assertEqual(Quaternion(a, b, c, d).conjugate(), Quaternion(a, -b, -c, -d))
    
    def test_double_conjugate(self):
        q = Quaternion.random()
        self.assertEqual(q, q.conjugate().conjugate())

    def test_norm(self):
        r = randomElements()
        q = Quaternion(*r)  
        self.assertEqual(q.norm(), np.linalg.norm(np.array(r)))

    def test_inverse(self):
        q = Quaternion.random()
        self.assertEqual(q * q.inverse(), Quaternion(1.0, 0.0, 0.0, 0.0))

    def test_versor(self): # normalise to unit quaternion
        pass

    def test_multiplicative_norm(self):
        q1 = Quaternion.random()
        q2 = Quaternion.random()
        self.assertAlmostEqual((q1 * q2).norm(), q1.norm() * q2.norm(), 7)

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


 # See https://github.com/erossignon/pyquaternion/blob/master/test_quaternion.py
 
if __name__ == '__main__':
    unittest.main()