#Object Initialisation

A Quaternion object can be created in the following ways:

## Default
> **`Quaternion()`**

Creates a unit quaternion `1 + 0i + 0j + 0k`: the quaternion representation of the real number 1.0.

		q1 = Quaternion()

For the purposes of rotation, this is a null quaternion (has no effect on the rotated vector).
For the purposes of quaternion multiplication, this is a unit quaternion (has no effect when multiplying)


## Copy
> **`Quaternion(other)`**

Clone another quaternion object

**Params:**

* `other` must be another Quaternion instance.

```
q2 = Quaternion(q1)
```

**Raises:** `TypeError` if the provided object is not an instance of Quaternion, or any valid positional argument as outlined below.

## Random 
> **`Quaternion.random()`**

Create a random quaternion that describes a rotation randomly chosen from a uniform distribution across the rotation space. [Source](http://planning.cs.uiuc.edu/node198.html).

This is a class method and is called as a method of the class itself rather than on a particular instance.
		
		q3 = Quaternion.random() # called as a class method
	
## From scalar
> **`Quaternion(scalar)`**

Create the quaternion representation of a scalar (single real number) value.

**Params:**

* `scalar` can be a real number, or a string representing a real number.

The imaginary part of the resulting quaternion will always be `0i + 0j + 0k`.

		q4 = Quaternion(4.7349)
		q4 = Quaternion(-3)
		q4 = Quaternion("4.7349")
		q4 = Quaternion("98")
		


**Raises:** 

* `TypeError` if the provided value cannot be converted to a real number.
* `ValueError` if a provided string cannot be interpreted as a real number.

## From elements
> **`Quaternion(w, x, y, z)`**

Create a quaternion by specifying 4 real-numbered scalar elements.

**Params:**

* `w, x, y, z` can be real numbers, strings representing real numbers, or a mixture of both.

```
q5 = Quaternion(1, 1, 0, 0)
q5 = Quaternion("1.0", "0", ""0.347"", "0.0")
q5 = Quaternion("1.76", 0, 0, 0)
```	

**Raises:** 

* `TypeError` if any of the provided values cannot be converted to a real number.
* `ValueError` if any of the provided strings cannot be interpreted as a real number.

## From a numpy array
> **`Quaternion(array)`**

Create a quaternion from the elements of a 4-element Numpy array

**Params:**

* `array` must be a 4-element numpy array containing real valued elements.

The elements `[a, b, c, d]` of the array correspond the the real, and each imaginary component respectively in the order `a + bi + cj + dk`.

		q6 = Quaternion(numpy.array([a, b, c, d]))
	
**Raises:** 

* `TypeError` if any of the array contents cannot be converted to a real number.
* `ValueError` if the array contains less/more than 4 elements
	
## From a sequence
> **`Quaternion(seq)`**

Create a quaternion object from an ordered sequence containing 4 real valued scalar elements

**Params:**

* `seq` can be a list, a tuple, a generator or any iterable sequence containing 4 values, each convertible to a real number.

The sequential elements `a, b, c, d` of the sequence correspond the the real, and each imaginary component respectively in the order `a + bi + cj + dk`.

		q7 = Quaternion((a, b, c, d)) // from 4-tuple
		q7 = Quaternion([a, b, c, d]) // from list of 4 
		

**Raises:** 

* `TypeError`  if any of the sequence contents cannot be converted to a real number.
* `ValueError` if the sequence contains less/more than 4 elements
	
## Using named parameters:

### Explicitly by element
> **`Quaternion(a=w, b=x, c=y, d=z)`**

Specify each element, using any sequence of ordered labels

**Params:**

* `a=w, b=x, c=y, d=z` can be real numbers, strings representing real numbers, or a mixture of both.

```
q8a = Quaternion(a=1.0, b=0.0, c=0.0, d=0.0)
q8a = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
q8a = Quaternion(a=1.0, i=0.0, j=0.0, k=0.0)
q8a = Quaternion(q1=1.0, q2=0.0, q3=0.0, q4=0.0)
```
		
**Rasises:** Exception behaviour is the same as initialisation by element as described above.

### Explicitly by component
> **`Quaternion(scalar=s, vector=v)` or `Quaternion(real=r, imaginary=i)`**

Specify the scalar (real) and vector (imaginary) parts of the desired quaternion.

**Params:**

* `scalar=s` or `real=r` can be a real number, or a string representing a real number.

* `vector=v`or `imaginary=i` can be a sequence or numpy array containing 3 real numbers.

Either component (but not both) may be absent, `None` or empty, and will be assumed to be zero in that case.
		
		q8b = Quaternion(scalar=1.0, vector=(0.0, 0.0, 0.0)) // Using 3-tuple
		q8b = Quaternion(scalar=None, vector=[1.0, 0.0, 0.0]) // Using list
		q8b = Quaternion(vector=numpy.array([1.0, 0.0, 0.0])) // Using Numpy 3-array
		
		q8b = Quaternion(real=1.0, imaginary=(0.0, 0.0, 0.0)) // Using 3-tuple
		q8b = Quaternion(real=None, imaginary=[1.0, 0.0, 0.0]) // Using list
		q8b = Quaternion(imaginary=numpy.array([1.0, 0.0, 0.0])) // Using Numpy 3-array
		
**Raises:** `ValueError` if the `vector` or `imaginary` component contains less/more than 3 elements

### Explicitly by rotation parameters
> **`Quaternion(axis=ax, angle=theta)`**

Specify the angle (in radians) for a rotation about an axis vector [x, y, z] to be described by the quaternion object.

**Params** 

* `angle=theta` can be a real number, or a string representing a real number.

* `axis=ax` can be a sequence or numpy array containing 3 real numbers. It can have any magnitude except `0`.

Both `axis` and `angle` must be provided to describe a meaningful rotation.

	
		q8c = Quaternion(angle=math.pi/2, axis=(1.0, 0.0, 0.0)) // Using 3-tuple
		q8c = Quaternion(angle=math.pi/2, axis=[1.0, 0.0, 0.0]) // Using list
		q8c = Quaternion(angle=math.pi/2, axis=numpy.array([1.0, 0.0, 0.0])) // Using Numpy 3-array
		
**Raises:** `

* `ValueError` if either `angle` or `axis` is missing
* `ValueError` if `axis` contains less/more than 3 elements  
* `ZeroDivisionError` if `axis` has 0 length.

### Explicitly by rotation or transformation matrix
> **`Quaternion(matrix=R)` or `Quaternion(matrix=T)`**

Specify the 3x3 rotation matrix (`R`) or 4x4 transformation matrix (`T`) from which the quaternion's rotation should be created. 

**Params:**

* `matrix=R` can be a 3x3 numpy array or matrix
* `matrix=T` can be a 4x4 numpy array or matrix. In this case, the translation part will be ignored, and only the rotational component of the matrix will be encoded within the quaternion. 

**Important:** The rotation component of the provided matrix must be a pure rotation i.e. [special orthogonal](http://mathworld.wolfram.com/SpecialOrthogonalMatrix.html).

	
		rotation = numpy.eye(3)
		transformation = numpy.eye(4)
		q8d = Quaternion(matrix=rotation) // Using 3x3 rotation matrix
		q8d = Quaternion(matrix=transformation) // Using 4x4 transformation matrix
 
This code uses a modification of the algorithm described in [Converting a Rotation Matrix to a Quaternion](https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf), which is itself based on the method described [here](http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/).

**Note:** Both matrices and quaternions avoid the singularities and discontinuities involved with rotation in 3 dimensions by adding extra dimensions. This has the effect that different values could represent the same rotation, for example quaternion q and -q represent the same rotation. It is therefore possible that, when converting a rotation sequence, the output may jump between these equivalent forms. This could cause problems where subsequent operations such as differentiation are done on this data. Programmers should be aware of this issue.
 
**Raises:** 

* `ValueError` if the matrix is not 3x3 or 4x4 or if the matrix is not special orthogonal.

* `TypeError` if the matrix is of the wrong type
 
### Explicitly by a numpy array
> **`Quaternion(array=a)`**

Specify a numpy 4-array of quaternion elements to be assigned directly to the internal vector representation of the quaternion object.

This is more direct, and may be faster than feeding a numpy array as a positional argument to the initialiser.

**Params:**

* `array=a` must be a 4-element numpy array containing real valued elements.

The elements `[a, b, c, d]` of the array correspond the the real, and each imaginary component respectively in the order `a + bi + cj + dk`.
		
		q8e = Quaternion(array=numpy.array([1.0, 0.0, 0.0, 0.0]))
		

**Raises:** `ValueError` if the array vector contains less/more than 4 elements
