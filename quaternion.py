# A python module for basic quaternion math

import numpy as np
import random
from math import sqrt, pi, sin, cos, tan

class Quaternion:

    def __init__(self, *args, **kwargs):
        s = len(args)
        if s is 0:
            # No positional arguments supplied
            if len(kwargs) > 0:
                # Keyword arguments provided
                if ("scalar" in kwargs) or ("vector" in kwargs):
                    scalar = kwargs.get("scalar", 0.0)
                    if scalar is None:
                        scalar = 0.0
                    else:
                        scalar = float(scalar)

                    vector = kwargs.get("vector", [])   
                    vector = self._validate_number_sequence(vector, 3)

                    self.q = np.hstack((scalar, vector)) 
                elif ("real" in kwargs) or ("imaginary" in kwargs):
                    real = kwargs.get("real", 0.0)
                    if real is None:
                        real = 0.0
                    else:
                        real = float(real)

                    imaginary = kwargs.get("imaginary", [])   
                    imaginary = self._validate_number_sequence(imaginary, 3)

                    self.q = np.hstack((real, imaginary))
                elif ("axis" in kwargs) or ("angle" in kwargs):
                    try:
                        axis = self._validate_number_sequence(kwargs["axis"], 3)
                        angle = float(kwargs["angle"])
                    except KeyError:
                        raise ValueError("Both 'axis' and 'angle' must be provided to describe a meaningful rotation.")
                    else:
                        self.q = Quaternion._from_axis_angle(axis, angle).q
                elif "array" in kwargs:
                    self.q = self._validate_number_sequence(kwargs["array"], 4)
                elif "matrix" in kwargs:
                    self.q = Quaternion._from_matrix(kwargs["matrix"]).q
                else:
                    keys = sorted(kwargs.keys())
                    elements = [kwargs[kw] for kw in keys]
                    if len(elements) is 1:
                        r = float(elements[0])
                        self.q = np.array([r, 0.0, 0.0, 0.0])
                    else:
                        self.q = self._validate_number_sequence(elements, 4)

            else: 
                # Default initialisation
                self.q = np.array([1.0, 0.0, 0.0, 0.0])
        elif s is 1:
            # Single positional argument supplied
            if isinstance(args[0], self.__class__):
                self.q = args[0].q
                return
            if args[0] is None:
                raise TypeError("Object cannot be initialised from " + str(type(args[0])))
            try:
                r = float(args[0])
                self.q = np.array([r, 0.0, 0.0, 0.0])
                return
            except(TypeError):
                pass # If the single argument is not scalar, it should be a sequence

            self.q = self._validate_number_sequence(args[0], 4)
            return
        
        else: 
            # More than one positional argument supplied
            self.q = self._validate_number_sequence(args, 4)

    def _validate_number_sequence(self, seq, n):
        if seq is None:
            return np.zeros(n)
        if len(seq) is n:
            try:
                l = [float(e) for e in seq]
            except ValueError:
                raise ValueError("One or more elements in sequence <" + str(seq) + "> cannot be interpreted as a real number")
            else:
                return np.array(l)
        elif len(seq) is 0:
            return np.zeros(n)
        else:
            raise ValueError("Unexpected number of elements in sequence.")

    # Initialise from matrix
    @classmethod
    def _from_matrix(cls, matrix):
        """ Create a Quaternion by specifying the 3x3 rotation or 4x4 transformation matrix 
        (as a numpy array) from which the quaternion's rotation should be created.

        """
        return cls()

    # Initialise from axis-angle
    @classmethod
    def _from_axis_angle(cls, axis, angle):
        """
        Precondition: axis is a valid numpy 3-vector, angle is a real valued angle in radians
        """
        mag_sq = np.dot(axis, axis)
        if mag_sq == 0.0:
            raise ZeroDivisionError("Provided rotation axis has no length")
        # Ensure axis is in unit vector form
        if (abs(1.0 - mag_sq) > 1e-14):
            axis = axis / sqrt(mag_sq)
        theta = angle / 2.0
        r = cos(theta)
        i = axis * sin(theta)

        return cls(r, i[0], i[1], i[2])


    

    @classmethod
    def random(cls):
        """ Generate a random unit quaternion. 

        Uniformly distributed across the rotation space
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

    # Representation
    def __str__(self):
        string = "{:.2f} {:+.2f}i {:+.2f}j {:+.2f}k".format(self.q[0], self.q[1], self.q[2], self.q[3])
        string += " (Axis: {} | Angle: {})".format(list(self.axis()), self.angle())
        return string

    def __repr__(self):
        return "{:.3f} {:+.3f}i {:+.3f}j {:+.3f}k".format(self.q[0], self.q[1], self.q[2], self.q[3])

    def __format__(self, formatstr):
        string = \
            "{:" + formatstr +"} "  + \
            "{:" + formatstr +"}i " + \
            "{:" + formatstr +"}j " + \
            "{:" + formatstr +"}k "
        return string.format(self.q[0], self.q[1], self.q[2], self.q[3])
        
    def __bool__(self):
        return not (self == Quaternion(0.0))

    def __nonzero__(self):
        return bool(self)

    # Comparison
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            tolerance = 1.0e-14
            try:
                isEqual = (abs(self.q - other.q) <= tolerance).all()
            except AttributeError:
                raise AttributeError("Error in internal quaternion representation means it cannot be compared like a numpy array.")
            return isEqual
        return self.__eq__(self.__class__(other))

    # Negation
    def __neg__(self):
        return self.__class__(array= -self.q)

    def __invert__(self):
        return self.inverse()

    # Addition
    def __add__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(array=self.q + other.q)
        return self + self.__class__(other)

    def __iadd__(self, other):
        return self + other

    def __radd__(self, other):
        return self + other

    # Subtraction
    def __sub__(self, other):
        return self + (-other)

    def __isub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -(self - other)

    # Multiplication
    def __mul__(self, other):
        """ Product of either:
        a) this quaternion, right-multiplied with another quaternion object, or
        b) this quaternion, scaled by a real-valued scalar.
        
        Returns a new quaternion object storing the result of the multiplication.
        Raises TypeError for incompatible operand types.
        """

        if isinstance(other, self.__class__):
            return self.__class__(array=np.dot(self._q_matrix(), other.q))
        return self * self.__class__(other)
    
    def __imul__(self, other):
        return self * other

    def __rmul__(self, other):
        return self.__class__(other) * self

    # Division
    def __div__(self, other):
        # Only implemented for scalar i.e. q / 2.3 (due to non-commutativity of quaternion multiplication) 
        try:
            scalar = 1.0 / float(other)
        except TypeError:
            raise TypeError("Quaternion division only defined for a scalar divisor")
        return self * self.__class__(scalar)

    def __idiv__(self, other):
        return self.__div__(other)

    def __rdiv__(self, other):
        return other * self.inverse()

    def __truediv__(self, other):
        return self.__div__(other)

    def __itruediv__(self, other):
        return self.__idiv__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    # Exponentiation
    def __pow__(self, exponent):
        exponent = int(exponent)
        if (exponent is 0):
            return Quaternion() # Unit quaternion
        elif (exponent >= 1): 
            q = Quaternion(self) # clone
            while (exponent > 1):
                q *= self
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
        return self.__class__(scalar=self.scalar(), vector= -self.vector())

    def inverse(self):
        return self.conjugate() / self._sum_of_squares()

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

    def magnitude(self):
        return self.norm()

    def _normalise(self):
        self.q = self.q / self.norm()

    def _fast_normalise(self):
        pass

    def normalised(self):
        """ Return a unit quaternion object (versor) representing the same rotation as this

        Result is guaranteed to be a unit quaternion
        """
        q = Quaternion(self)
        q._normalise()
        return q

    def versor(self):
        """ Return a unit quaternion object (versor) representing the same rotation as this

        Result is guaranteed to be a unit quaternion
        """
        return self.normalised()

    def _q_matrix(self):
        return np.array([
            [self.q[0], -self.q[1], -self.q[2], -self.q[3]],
            [self.q[1],  self.q[0], -self.q[3],  self.q[2]],
            [self.q[2],  self.q[3],  self.q[0], -self.q[1]],
            [self.q[3], -self.q[2],  self.q[1],  self.q[0]]])

    def _q_bar_matrix(self):
        return np.array([
            [self.q[0], -self.q[1], -self.q[2], -self.q[3]],
            [self.q[1],  self.q[0],  self.q[3], -self.q[2]],
            [self.q[2], -self.q[3],  self.q[0],  self.q[1]],
            [self.q[3],  self.q[2], -self.q[1],  self.q[0]]])

    def _rotate_quaternion(self, q):
        """ Rotate a quaternion vector using the stored rotation.

        The input q is the vector to be rotated, in quaternion form (0 + xi + yj + kz)
        """
        self._normalise()
        return self * q * self.conjugate()

    def rotate(self, vector):
        """ Rotate a vector using the quaternion's stored rotation (similarity transform)

        Input vector can be specified as another (pure imaginary) quaternion 
        or a 3-vector described by a tuple, list or numpy array of length 3.

        Output is returned in the type of the provided input vector
        """
        if isinstance(vector, self.__class__):
            return self._rotate_quaternion(vector)
        q = Quaternion(vector=vector)
        a = self._rotate_quaternion(q).vector()
        l = [x for x in a]
        if isinstance(vector, list):
            return l
        elif isinstance(vector, tuple):
            return tuple(l)
        else:
            return a

    def rotation_matrix(self):
        self._normalise()
        product_matrix = np.dot(self._q_matrix(), self._q_bar_matrix().conj().transpose())
        return product_matrix[1:][:,1:]

    def transformation_matrix(self):
        t = np.array([[0.0], [0.0], [0.0]])
        Rt = np.hstack([self.rotation_matrix(), t])
        return np.vstack([Rt, np.array([0.0, 0.0, 0.0, 1.0])])

    def axis(self):
        #TODO
        return np.array([0.0, 0.0, 0.0])

    def angle(self):
        #TODO
        return 0.0

    def scalar(self):
        """ Return the real or scalar component of the quaternion object

        Result is a real number i.e. float
        """
        return self.q[0]

    def vector(self):
        """ Return the imaginary or vector component of the quaternion object

        Result is a numpy 3-array of floats
        Result is not guaranteed to be a unit vector
        """
        return self.q[1:4]

    def real(self):
        """ Return the real or scalar component of the quaternion object

        Result is a real number i.e. float
        """
        return self.scalar()

    def imaginary(self):
        """ Return the imaginary or vector component of the quaternion object

        Result is a numpy 3-array of floats
        Result is not guaranteed to be a unit vector
        """
        return self.vector()

    def elements(self):
        """ Return all the elements of the quaternion object

        Result is a numpy 4-array of floats
        Result is not guaranteed to be a unit vector
        """
        return self.q

    def __deepcopy__(self):
        return self.__class__(self)
