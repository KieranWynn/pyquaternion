# Unit tests for quaternion module

import unittest
import numpy as np
from quaternion import Quaternion
 
class TestQuaternionInitialisation(unittest.TestCase):
 
    def setUp(self):
        pass

    # Test initialisation of objects
    def test_init(self):
        q = Quaternion(np.array([1., 0., 0., 0.]))
        self.assertIsInstance(q, Quaternion)

    def test_init_incorrect_type(self):
        self.assertRaises(TypeError, Quaternion, "1, 2, 3, 4")

    def test_init_incorrect_size(self):
        self.assertRaises(RuntimeError, Quaternion, np.array([1., 0., 0.]))

    def test_init_default(self):
        q = Quaternion()
        self.assertIsInstance(q, Quaternion)

    def test_init_from_array_correct(self):
        q = Quaternion.from_array(np.array([1., 0., 0., 0.]))
        self.assertIsInstance(q, Quaternion)

    def test_init_from_array_incorrect_type(self):
        self.assertRaises(TypeError, Quaternion.from_array, 3.0)
 
    def test_init_from_elements_correct(self):
        q = Quaternion.from_elements(1., 0., 0., 0.)
        self.assertIsInstance(q, Quaternion)

    def test_init_from_elements_incorrect_type(self):
        self.assertRaises(TypeError, Quaternion.from_elements, "1", "0", "0", "0")

class TestQuaternionFeatures(unittest.TestCase):
 
    def setUp(self):
        pass

    def test_equality(self):
        self.assertEqual(Quaternion.from_elements(1, 2, 4, 5), Quaternion.from_elements(1, 2, 4, 5))

    def test_equality_false(self):
        self.assertNotEqual(Quaternion.from_elements(1, 2, 3, 4), Quaternion.from_elements(1, 2, 3, 5))

    def test_conjugate(self):
        self.assertEqual(Quaternion.from_elements(1, 2, 4, 5).conjugate(), Quaternion.from_elements(1, -2, -4, -5))

 # See https://github.com/erossignon/pyquaternion/blob/master/test_quaternion.py
 
if __name__ == '__main__':
    unittest.main()