# A python module for basic quaternion math

import numpy as np

class Quaternion:

	# Initialise with 'none' (default)
	def __init__(self, q = None):
		if q is None:
			# q is a 1D vector of the elements of the quaternion
			self.q = np.array([1., 0., 0., 0.])
		else:
			try:
				if (q.size is not 4):
					raise RuntimeError("Incorrect array size: Input to initialiser must be a 1D array of length 4")
			except AttributeError:
				raise TypeError("Incorrect input type: Input to initialiser must be a 1D array of length 4")
			self.q = q

	# Initialise with array
	@classmethod
	def from_array(cls, array):
		return cls(array)

	# Initialise from elements
	@classmethod
	def from_elements(cls, a, b, c, d):
		return cls(np.array([a, b, c, d]))

	# Initialise from axis-angle

	# Initialise from rotation matrix

	def conjugate(self):
		# Construct conjugate array
		conj = np.hstack((self.q[1], -self.q[1:4]))
		# Return conjugate encapsulated in a new instance
		return self.__class__(conj)

	def norm(self):
		# Return type: Scalar (double)
		# return sqrt(dot(self.q,self.q))
		return ( 
			self.q[0]**2 + 
			self.q[1]**2 + 
			self.q[2]**2 + 
			self.q[3]**2 )

	def inverse(self):
		return self.conjugate() / self.norm()

	def product(self, other):
		pass
		#self.q[0] * other.q[0]

	def q_matrix(self):
		pass

	def real(self):
		return self.q[0]

	def imaginary(self):
		return self.q[1:4]

	def normalise(self):
		pass

	def __eq__(self, other):
		self.normalise()
		other.normalise()
		return (self.q == other.q).all()




