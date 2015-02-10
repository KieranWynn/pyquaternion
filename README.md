#pyquaternion

A barebones python module for quaternion manipulation and 3D rotation


**Designed for Python 3.0+**


> 	*** IMPORTANT NOTE ***
> 
> 	This module is still under development, and is, at the moment, incomplete.
> 
> 	See test output for currently implemented features:
> 	
> 	python3 test_quaternion.py -v



## Initialisation
A Quaternion object can be created in the following ways:

1. Default - Unit quaternion (1 + 0i + 0j + 0k)

		q1 = Quaternion()
	Creates a unit quaternion, the quaternion representation of 1.0.
	For the purposes of rotation, this is a null quaternion (has no effect when rotating).
	For the purposes of quaternion multiplication, this is a unit quaternion (has no effect when multiplying)
	
2. Copy - Clone another quaternion object

		q2 = Quaternion(q1)
	> Raises *TypeError* if the provided object is not an instance of Quaternion, or any valid positional argument as outlined below.

3. Random - Create a random quaternion describing a rotation randomly chosen from a uniform distribution across the rotation space.
		
		q3 = Quaternion.random()
	
4. From scalar - Create the quaternion representation of a scalar (single real number) value.

		q4 = Quaternion(4.7349)
		q4 = Quaternion(-3)
		q4 = Quaternion("4.7349")
		q4 = Quaternion("98")
		
	The imaginary part of the resulting quaternion will always be [0.0i, 0.0j, 0.0k]
	The scalar can be a real number, or a string representing a real number.	
	> Raises *TypeError* if the provided value cannot be converted to a real number.
	>
	> Raises *ValueError* if a provided string cannot be interpreted as a real number.

5. From elements - Create a quaternion by specifying 4 real-numbered scalar elements.

		q5 = Quaternion(1, 1, 0, 0)
		q5 = Quaternion("1.0", "0", ""0.347"", "0.0")
		q5 = Quaternion("1.76", 0, 0, 0)
		
	Elements can be real numbers, strings representing real numbers, or a mixture of both.
	> Raises *TypeError* if any of the provided values cannot be converted to a real number.
	>
	> Raises *ValueError* if any of the provided strings cannot be interpreted as a real number.

4. From a Numpy array - Create a quaternion from the elements of a 4-element Numpy array

		q5 = Quaternion(numpy.array([a, b, c, d]))
	The elements a, b, c, d of the array correspond the the real, and each imaginary component respectively in the order a + bi + cj + dk.
	
	> Raises *TypeError* if any of the array contents cannot be converted to a real number.
	>
	> Raises *ValueError* if the array contains less/more than 4 elements
	
5. From a sequence - Create a quaternion object from a sequence containing 4 real valued scalar elements

		q6 = Quaternion((a, b, c, d)) // from 4-tuple
		q6 = Quaternion([a, b, c, d]) // from list of 4 
		
	The sequential elements a, b, c, d of the sequence correspond the the real, and each imaginary component respectively in the order a + bi + cj + dk.
	
	> Raises *TypeError* if any of the sequence contents cannot be converted to a real number.
	>
	> Raises *ValueError* if the sequence contains less/more than 4 elements
	
6. Using named parameters:
	a.	Explicitly by element - specify each element, using any sequence of ordered labels
			
			q7a = Quaternion(a=1.0, b=0.0, c=0.0, d=0.0)
			q7a = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
			q7a = Quaternion(a=1.0, i=0.0, j=0.0, k=0.0)
			q7a = Quaternion(q1=1.0, q2=0.0, q3=0.0, q4=0.0)
	> Exception behaviour is the same as initialisation by element as described above.
	
	b. Explicitly by component - specify the scalar (real) and vector (imaginary) parts of the desired quaternion.
			
			q7b = Quaternion(scalar=1.0, vector=(0.0, 0.0, 0.0)) // Using 3-tuple
			q7b = Quaternion(scalar=1.0, vector=[0.0, 0.0, 0.0]) // Using list
			q7b = Quaternion(scalar=1.0, vector=numpy.array([0.0, 0.0, 0.0])) // Using Numpy 3-array
			
			q7b = Quaternion(real=1.0, imaginary=(0.0, 0.0, 0.0)) // Using 3-tuple
			q7b = Quaternion(real=1.0, imaginary=[0.0, 0.0, 0.0]) // Using list
			q7b = Quaternion(real=1.0, imaginary=numpy.array([0.0, 0.0, 0.0])) // Using Numpy 3-array
	Either of ( *scalar* | *real* ) and ( *vector* | *imaginary* ) may be absent, *'None'* or empty, and will be assumed to be zero in that case.
	> Raises *ValueError* if the vector/imaginary component contains less/more than 3 elements
	c. Explicitly by rotation parameters - specify the angle (in radians) for a rotation about an axis vector [x, y, z] to be described by the quaternion object.
		
			q7c = Quaternion(angle=math.pi/2, axis=(1.0, 0.0, 0.0)) // Using 3-tuple
			q7c = Quaternion(angle=math.pi/2, axis=[1.0, 0.0, 0.0]) // Using list
			q7c = Quaternion(angle=math.pi/2, axis=numpy.array([1.0, 0.0, 0.0])) // Using Numpy 3-array
	> Raises *ValueError* if either *angle* or *axis* is missing - both must be provided to describe a meaningful rotation.
	>
	> Raises *ValueError* if the axis vector contains less/more than 3 elements 
	> 
	> Raises *ZeroDivisionError* if the axis vector has 0 length.
	
	d. Explicitly by rotation or transformation matrix - specify the 3x3 rotation or 4x4 transformation matrix (as a numpy array) from which the quaternion's rotation should be created.
			
			rotation = numpy.eye(3)
			transformation = numpy.eye(4)
			q7d = Quaternion(matrix=rotation) // Using 3x3 rotation matrix
			q7d = Quaternion(matrix=transformation) // Using 4x4 transformation matrix
	 Note: when using a transformation matrix as a basis, the translation part will be ignored, and only the rotational component of the matrix will be encoded within the quaternion.
	> Raises *ValueError* if the matrix is not 3x3 or 4x4.
	>
	> Raises *TypeError* if the matrix is of the wrong type
	 
	e. Explicitly by a numpy array  - specify a numpy 4-array of quaternion elements to be assigned directly to the internal vector representation of the quaternion object.
			
			q7e = Quaternion(array=numpy.array([1.0, 0.0, 0.0, 0.0]))
	This is more direct, and may be faster than feeding a numpy array as a positional argument to the initialiser.
	> Raises *ValueError* if the array vector contains less/more than 4 elements
	