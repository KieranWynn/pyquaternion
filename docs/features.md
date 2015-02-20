This page defines features available for pyquaternion's Quaternion objects

The code examples below assume the existence of a Quaternion object. You can recreate this by running the following in your Python interpreter of choice:
	
	my_quaternion = Quaternion.random()

# Norm
`Quaternion.norm()` & `Quaternion.magnitude()`

L2 norm of the quaternion 4-vector 

This should be 1.0 for a unit quaternion (versor)

**Returns:** a scalar real number representing the square root of the sum of the squares of the elements of the quaternion.

	my_quaternion.norm()
	my_quaternion.magnitude()
	
# Inversion
`Quaternion.inverse()`

Inverse of the quaternion object

For a unit quaternion, this is the inverse rotation, i.e. when combined with the original rotation, will result in the null rotation.

**Returns:** a new Quaternion object representing the inverse of this object
	
	inv_quaternion = my_quaternion.inverse()
	
# Conjugation
`Quaternion.conjugate()`

Quaternion conjugate

For a unit quaternion, this is the same as the inverse.

**Returns:** a new Quaternion object clone with its vector part negated

	conj_quaternion = my_quaternion.conjugate()
	
# Normalisation
`Quaternion.normalised()` & `Quaternion.versor()`
Get a unit quaternion (versor) version of this Quaternion object.

A unit quaternion (versor) has a norm() of 1.0

**Returns:** a new Quaternion object clone that is guaranteed to be a unit quaternion
	
	unit_quaternion = my_quaternion.normalise()
	unit_quaternion = my_quaternion.versor()
	

# Rotation
`Quaternion.rotate(vector)`
Rotate a 3D vector by the rotation stored in the Quaternion object

**Params:**
	
* **vector** - a 3-vector specified as any ordered sequence of 3 real numbers corresponding to x, y, and z values. Some types that are recognised are: numpy arrays, lists and tuples. A 3-vector can also be represented by a Quaternion object who's scalar part is 0 and vector part is the required 3-vector. Thus it is possible to call `Quaternion.rotate(q)` with another quaternion object as an input.

**Returns:** the rotated vector returned as the same type it was specified at input.

	rotated_tuple 		= my_quaternion.rotate((1, 0, 0)) # Returns a tuple
	rotated_list  		= my_quaternion.rotate([1.0, 0.0, 0.0]) # Returns a list
	rotated_array 		= my_quaternion.rotate(numpy.array([1.0, 0.0, 0.0])) # Returns a Numpy 3-array
	rotated_quaternion	= my_quaternion.rotate(Quaternion(vector=[1, 0, 0])) # Returns a Quaternion object

> Raises `TypeError` if any of the vector elements cannot be converted to a real number.
>
> Raises `ValueError` if **vector** cannot be interpreted as a 3-vector or a Quaternion object.


# Interpolation
...coming soon [Source](http://number-none.com/product/Understanding%20Slerp,%20Then%20Not%20Using%20It/)

# Conversion to matrix form
`Quaternion.rotation_matrix()` & `Quaternion.transformation_matrix()`

Get the 3x3 rotation or 4x4 homogeneous transformation matrix equivalent of the quaternion rotation.

**Returns:**

* `Quaternion.rotation_matrix()` : a 3x3 orthogonal rotation matrix as a 3x3 Numpy array
* `Quaternion.transformation_matrix()` : a 4x4 homogeneous transformation matrix as a 4x4 Numpy array

> **Note:** This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.
	
	R = my_quaternion.rotation_matrix() 		# 3x3 rotation matrix
	T = my_quaternion.transformation_matrix()   # 4x4 transformation matrix

# Accessing rotation axis
`Quaternion.axis()`

Get the axis or vector about which the quaternion rotation occurs

**Returns:** a Numpy unit 3-vector describing the Quaternion object's axis of rotation.

> **Note:** This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.

	u = my_quaternion.axis() # Unit vector about which rotation occurs


# Accessing rotation angle 
`Quaternion.angle()` 

Get the angle (in radians) describing the magnitude of the quaternion rotation about its rotation axis. This is guaranteed to be within the range (-pi:pi) with the direction of rotation indicated by the sign.

**Returns:** a real number in the range (-pi:pi) describing the angle of rotation in radians about a Quaternion object's axis of rotation. 

> **Note:** This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.

	theta = my_quaternion.angle() # Magnitude of rotation about the prescribed axis


# Accessing real components
`Quaternion.scalar()` & `Quaternion.real()`

Get the real or scalar component of the Quaternion object

> A quaternion can be described in terms of a scalar and vector part, q = [r, **v**] where:

> r is the scalar coefficient of the real part of the quaternion i.e. **a** in [a + b*i* + c*j* + d*k*]

> **v** is the 3-vector of coefficients to the imaginary parts of the quaternion i.e. **[b, c, d]** in [a + b*i* + c*j* + d*k*]

This method returns r

**Returns** the scalar, real valued element of the Quaternion object

	r = my_quaternion.scalar()
	r = my_quaternion.real()

# Accessing imaginary components
`Quaternion.vector()` & `Quaternion.imaginary()`

Get the imaginary or vector component of the Quaternion object. This can be used, for example, to extract the stored vector when a pure-imaginary quaternion object is used to describe a vector within the three-dimensional vector space.

> A quaternion can be described in terms of a scalar and vector part, q = [r, **v**] where:

> r is the scalar coefficient of the real part of the quaternion i.e. **a** in [a + b*i* + c*j* + d*k*]

> **v** is the 3-vector of coefficients to the imaginary parts of the quaternion i.e. **[b, c, d]** in [a + b*i* + c*j* + d*k*]

This method returns **v**

**Returns** Numpy 3-array of the 3 imaginary elements of the Quaternion object

	v = my_quaternion.vector()
	v = my_quaternion.imaginary()

# Accessing individual elements
`Quaternion.elements()`

Return all four elements of the quaternion object. Result is not guaranteed to be a unit 4-vector.

**Returns:** a numpy 4-array of real numbered coefficients.

	a = my_quaternion.elements()
	print("{} + {}i + {}j + {}k".format(a[0], a[1], a[2], a[3]))
