# A python module for basic quaternion math

import numpy as np
import random
from math import sqrt, pi, sin, cos, tan

class Quaternion:

    # Initialise with 'none' (default)
    def __init__(self, w = 1., x = 0., y = 0., z = 0.):
        # q is a 1D vector of the elements of the quaternion
        try:
            a = np.array([float(w), float(x), float(y), float(z)])
        except TypeError:
            # w is the lone parameter
            if isinstance(w, dict):
                a = np.array([w[k] for k in sorted(w.keys())])
            else:
                a = np.array(w) 
        self.q = a 

    # Initialise with array
    @classmethod
    def from_array(cls, array):
        """ accept input as array, list, tuple, dict
        """
        try:
            if (array.size is not 4):
                raise RuntimeError("Incorrect array size: Input to initialiser must be a 1D array of length 4")
        except AttributeError:
            raise TypeError("Incorrect input type: Input to initialiser must be a 1D array of length 4")
        
        instance = cls()
        instance.q = array
        return instance

    # Initialise from elements
    @classmethod
    def from_elements(cls, a, b, c, d):
        return cls(a, b, c, d)

    # Initialise from axis-angle

    # Initialise from rotation matrix

    @classmethod
    def random(cls):
        """ Generate a random unit quaternion.

        As per: http://planning.cs.uiuc.edu/node198.html
        """
        r1 = random.random()
        r2 = random.random()
        r3 = random.random()

        q1 = sqrt(1.0 - r1) * (sin(2 * pi * r2))
        q2 = sqrt(1.0 - r1) * (cos(2 * pi * r2))
        q3 = sqrt(r1)       * (sin(2 * pi * r3))
        q4 = sqrt(r1)       * (cos(2 * pi * r3))

        return cls(q1, q2, q3, q4)

    def __str__(self):
        string = "{:.2f} {:+.2f}i {:+.2f}j {:+.2f}k".format(self.q[0], self.q[1], self.q[2], self.q[3])
        string += " (Axis: {} | Angle: {})".format(self.axis(), self.angle())
        return string

    def __repr__(self):
        return "{:.3f} {:+.3f}i {:+.3f}j {:+.3f}k".format(self.q[0], self.q[1], self.q[2], self.q[3])

    def __eq__(self, other):
        #self.normalise()
        #other.normalise()
        tolerance = 1.0e-7
        try:
            isEqual = (abs(self.q - other.q) <= tolerance).all()
        except AttributeError:
            raise AttributeError("Internal quaternion representattion is not a numpy array and cannot be compared like one.")
        
        return isEqual

    def __neg__(self):
        return self.__class__.from_array(-self.q)

    def __add__(self, other):
        return self.__class__.from_array(self.q + other.q)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__class__.from_array(self.q - other.q)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if (type(self) is type(other)): # TODO This is wrong
            return self.__class__.from_array(np.dot(self.q_matrix(), other.q))
        else:
            copy = self.clone()
            copy._scale(other)
            return copy

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Only implemented for scalar (i.e. q / 2.3)
        try:
            scale = 1.0 / other
        except TypeError:
            raise
        else:
            copy = self.clone()
            copy._scale(scale)
            return copy

    def __pow__(self, exponent):
        if (exponent is 0):
            return Quaternion() # Unit quaternion
        elif (exponent >= 1): 
            q = self.clone()
            while (exponent > 1):
                q = q * self
                exponent -= 1
            return q
        else:
            # TODO: implement root behaviour (exponent < 1)
            pass

    def _vector_conjugate(self):
        return np.hstack((self.q[0], -self.q[1:4]))

    def _sum_of_squares(self):
        return np.dot(self.q, self.q)

    def conjugate(self):
        # Return vector conjugate encapsulated in a new instance
        return self.__class__.from_array(self._vector_conjugate())

    def norm(self): # -> scalar double
        """ Return L2 norm of the quaternion 4-vector 

        Returns the square root of the sum of the squares of the elements of q
        """
        # Optimised by using the magic number method described here: http://stackoverflow.com/a/12934750

        mag_squared = self._sum_of_squares()
        if (abs(1.0 - mag_squared) < 2.107342e-08):
            return ((1.0 + mag_squared) / 2.0) # More efficient. Padé approximation valid if error is small 
        else:
            return sqrt(mag_squared) # Error is too big, take the performance hit to calculate the square root properly

    def inverse(self):
        return self.__class__.from_array(self._vector_conjugate() / self._sum_of_squares())

    def q_matrix(self):
        return np.array([
            [self.q[0], -self.q[1], -self.q[2], -self.q[3]],
            [self.q[1],  self.q[0], -self.q[3],  self.q[2]],
            [self.q[2],  self.q[3],  self.q[0], -self.q[1]],
            [self.q[3], -self.q[2],  self.q[1],  self.q[0]]])

    def q_bar_matrix(self):
        return np.array([
            [self.q[0], -self.q[1], -self.q[2], -self.q[3]],
            [self.q[1],  self.q[0],  self.q[3], -self.q[2]],
            [self.q[2], -self.q[3],  self.q[0],  self.q[1]],
            [self.q[3],  self.q[2], -self.q[1],  self.q[0]]])

    def real(self):
        return self.q[0]

    def imaginary(self):
        return self.q[1:4]

    def normalised(self):
        q = self.clone()
        q._normalise()
        return q

    def magnitude(self):
        return self.norm()

    def axis(self):
        #TODO
        return np.array([0.0, 0.0, 0.0])

    def angle(self):
        #TODO
        return 0.0

    def versor(self):
        return self.normalise()

    def __deepcopy__(self):
        return self.__class__.from_array(self.q)

    def clone(self):
        return self.__deepcopy__()

    def _normalise(self):
        self.q = self.q / self.norm()

    def _scale(self, scalar):
        """ Apply a scalar multiple to the internal quaternion 4-vector.
        """
        try:
            s = float(scalar)
        except:
            raise
        else:
            self.q = self.q * s
        


    

