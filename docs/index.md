# Welcome
Welcome! pyquaternion is a full-featured Python module for representing and using quaternions.

The following should get you up and running with pyquaternion in no time.

<a name="getting_started"></a>
# Getting started
The following aims to familiarize you with the basic functionality of quaternions in pyquaternion. It provides an entry point and a quick orientation (no pun intended) for those who want get stuck straight in. More comprehensive feature summaries can be found in the [features](#quaternion-features) and [operations](#quaternion-operations) documentation.

If you want to learn more about quaternions and how they apply to certain problems, you can read about quaternions [here](http://en.wikipedia.org/wiki/Quaternion), and their application to rotations [here](http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation).

## Installation
To start, you will need to install *pyquaternion* into your environment.

Go into the pyquaternion repository root directory (the one with setup.py and README.md):

    $ cd <path_to_repo>

> [Optional] If you are using virtual environments, switch to or create your environment of choice now:

    $ workon <my_environment>

Now use pip to install *pyquaternion* and its dependencies

    $ pip install .

> Note: pyquaternion requires [Numpy](http://www.numpy.org) for the representation of arrays and matrices.
Chances are if you're needing quaternions, you've been dealing with numerical computation and you're already familiar with numpy.
If not, don't worry, it will be installed into your environment automatically.

Great, you now have pyquaternion installed and you're ready to roll. Or pitch. Or yaw. No judging here :)

## Basic Usage

In your code, simply import the *Quaternion* object from the *pyquaternion* module:

	>>> from pyquaternion import Quaternion

Next, create a Quaternion object to describe your desired rotation:

	>>> my_quaternion = Quaternion(axis=[1, 0, 0], angle=3.14159265)
> Note: There are many ways to create a Quaternion object. See the [initialisation](#object-initialisation) section for a complete guide.

Now you can do a bunch of useful things with your new quaternion object. Let's try rotating a vector:

	>>> import numpy
	>>> numpy.set_printoptions(suppress=True) # Suppress insignificant values for clarity
	>>> v = numpy.array([0., 0., 1.]) # Unit vector in the +z direction
	>>> v_prime = my_quaternion.rotate(q)
	>>> v_prime
		array([ 0., 0., -1.])
	>>>

A cool feature of quaternions is that they can be intuitively chained together to form a composite rotation from a sequence of discrete rotations:

	>>> q1 = Quaternion(axis=[1, 0, 0], angle=3.14159265) # Rotate 180 about X
	>>> q2 = Quaternion(axis=[0, 1, 0], angle=3.14159265 / 2) # Rotate 90 about Y
	>>> q3 = q1 * q2 # Composite rotation of q1 then q2 expressed as standard multiplication
	>>> v_prime = q3.rotate(v)
	>>> v_prime
		array([ 1., 0., 0.])
	>>>

Quaternions are used extensively in animation to describe smooth transitions between known orientations. This is known as interpolation. This is an example of an area where quaternions are preferred to rotation matrices as smooth interpolation is not possible with the latter. Here's quaternion interpolation in action:

	>>> import numpy
	>>> numpy.set_printoptions(suppress=True) # Suppress insignificant values for clarity
	>>> v = numpy.array([0., 0., 1.]) # Unit vector in the +z direction
	>>> q0 = Quaternion(axis=[1, 1, 1], angle=0.0) # Rotate 0 about x=y=z
	>>> q1 = Quaternion(axis=[1, 1, 1], angle=2 * 3.14159265 / 3) # Rotate 120 about x=y=z
	>>> for q in Quaternion.intermediates(q0, q1, 8, include_endpoints=True):
	...		v_prime = q.rotate(v)
	...		print(v_prime)
	...
	[ 0.  0.  1.]
	[ 0.14213118 -0.12416109  0.98202991]
	[ 0.29457011 -0.22365854  0.92908843]
	[ 0.44909878 -0.29312841  0.84402963]
	[ 0.59738651 -0.32882557  0.73143906]
	[ 0.73143906 -0.32882557  0.59738651]
	[ 0.84402963 -0.29312841  0.44909879]
	[ 0.92908843 -0.22365854  0.29457012]
	[ 0.98202991 -0.12416109  0.14213118]
	[ 1. 0.  0.]

In the code above, the expression `Quaternion.intermediates(q0, q1, 8, include_endpoints=True)` returns an iterator over a sequence of Quaternion objects describing a set of 10 (8 + 2) rotations between `q0` and `q1`. The printed output is then the path of the point originally at [0, 0, 1] as it is rotated through 120 degrees about x=y=z to end up at [1, 0, 0].
This could easily be plugged into a visualisation framework to show smooth animated rotation sequences. Read the full documentation on interpolation features [here](#quaternion-features).

For a full demonstration of 3D interpolation and animation, run the `demo.py` script included in the pyquaternion package. This will require some elements of the full [SciPy](http://www.scipy.org/about.html) package that are not required for pyquaternion itself.


# Object Initialisation

A Quaternion object can be created in the following ways:

## Default
> **`Quaternion()`**

Creates a unit quaternion `1 + 0i + 0j + 0k`: the quaternion representation of the real number 1.0, 
and the representation of a null rotation.

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


## Explicitly by element
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

## Explicitly by component
> **`Quaternion(scalar=s, vector=v)` or `Quaternion(real=r, imaginary=i)`**

Specify the scalar (real) and vector (imaginary) parts of the desired quaternion.

**Params:**

* `scalar=s` or `real=r` can be a real number, or a string representing a real number.
* `vector=v` or `imaginary=i` can be a sequence or numpy array containing 3 real numbers.

Either component (but not both) may be absent, `None` or empty, and will be assumed to be zero in that case.

    q8b = Quaternion(scalar=1.0, vector=(0.0, 0.0, 0.0)) // Using 3-tuple
    q8b = Quaternion(scalar=None, vector=[1.0, 0.0, 0.0]) // Using list
    q8b = Quaternion(vector=numpy.array([1.0, 0.0, 0.0])) // Using Numpy 3-array

    q8b = Quaternion(real=1.0, imaginary=(0.0, 0.0, 0.0)) // Using 3-tuple
    q8b = Quaternion(real=None, imaginary=[1.0, 0.0, 0.0]) // Using list
    q8b = Quaternion(imaginary=numpy.array([1.0, 0.0, 0.0])) // Using Numpy 3-array

**Raises:** `ValueError` if the `vector` or `imaginary` component contains less/more than 3 elements

## Explicitly by rotation parameters
> **`Quaternion(axis=ax, radians=rad)`** or **`Quaternion(axis=ax, degrees=deg)`** or **`Quaternion(axis=ax, angle=theta)`**

Specify the angle (qualified as radians or degrees) for a rotation about an axis vector [x, y, z] to be described by the quaternion object.

**Params**
* `axis=ax` can be a sequence or numpy array containing 3 real numbers. It can have any magnitude except `0`.
* `radians=rad` [optional] a real number, or a string representing a real number in radians.
* `degrees=deg` [optional] a real number, or a string representing a real number in degrees.
* `angle=theta` [optional] a real number, or a string representing a real number in radians.

The `angle` (radians/degrees/angle) keyword may be absent, `None` or empty, and will be assumed to be zero in that case,
but the `axis` keyword must be provided to describe a meaningful rotation.


    q8c = Quaternion(axis=(1.0, 0.0, 0.0), radians=math.pi/2) // Using radians asnd a 3-tuple
    q8c = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90) // Using degrees and a list
    q8c = Quaternion(axis=numpy.array([1.0, 0.0, 0.0]), angle=math.pi/2) // Using radians and a Numpy 3-array

**Raises:**

* `ValueError` if `axis` is missing
* `ValueError` if `axis` contains less/more than 3 elements
* `TypeError` if `radians/degrees/angle` cannot be interpreted as a real number
* `ZeroDivisionError` if `axis` has 0 length.

## Explicitly by rotation or transformation matrix
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

## Explicitly by a numpy array
> **`Quaternion(array=a)`**

Specify a numpy 4-array of quaternion elements to be assigned directly to the internal vector representation of the quaternion object.

This is more direct, and may be faster than feeding a numpy array as a positional argument to the initialiser.

**Params:**

* `array=a` must be a 4-element numpy array containing real valued elements.

The elements `[a, b, c, d]` of the array correspond the the real, and each imaginary component respectively in the order `a + bi + cj + dk`.

    q8e = Quaternion(array=numpy.array([1.0, 0.0, 0.0, 0.0]))


**Raises:** `ValueError` if the array vector contains less/more than 4 elements


# Quaternion Features
This section defines features available for pyquaternion's Quaternion objects

The code examples below assume the existence of a Quaternion object. You can recreate this by running the following in your Python interpreter of choice:

	my_quaternion = Quaternion.random()

## Norm
> **`norm` or `magnitude`**

L2 norm of the quaternion 4-vector

This should be 1.0 for a unit quaternion (versor)

**Returns:** a scalar real number representing the square root of the sum of the squares of the elements of the quaternion.

	my_quaternion.norm
	my_quaternion.magnitude

> **`is_unit(tolerance=1e-14)`**

**Params:**

* `tolerance` - [optional] - maximum absolute value by which the norm can differ from 1.0 for the object to be considered a unit quaternion. Defaults to `1e-14`.

**Returns:** `True` if the Quaternion object is of unit length to within the specified tolerance value. `False` otherwise.


## Inversion
> **`inverse`**

Inverse of the quaternion object

For a unit quaternion, this is the inverse rotation, i.e. when combined with the original rotation, will result in the null rotation.

**Returns:** a new Quaternion object representing the inverse of this object

	inv_quaternion = my_quaternion.inverse

## Conjugation
> **`conjugate`**

Quaternion conjugate

For a unit quaternion, this is the same as the inverse.

**Returns:** a new Quaternion object clone with its vector part negated

	conj_quaternion = my_quaternion.conjugate

## Normalisation
> **`normalised` or `unit`**

Get a unit quaternion (versor) copy of this Quaternion object.

A unit quaternion has a `norm` of 1.0

**Note:** A Quaternion representing zero i.e. `Quaternion(0, 0, 0, 0)` cannot be normalised. In this case, the returned object will remain zero.

**Returns:** a new Quaternion object clone that is guaranteed to be a unit quaternion *unless* the original object was zero, in which case the norm will remain zero.

	unit_quaternion = my_quaternion.normalised
	unit_quaternion = my_quaternion.unit


## Rotation
> **`rotate(vector)`**

Rotate a 3D vector by the rotation stored in the Quaternion object

**Params:**

* `vector` - a 3-vector specified as any ordered sequence of 3 real numbers corresponding to x, y, and z values. Some types that are recognised are: numpy arrays, lists and tuples. A 3-vector can also be represented by a Quaternion object who's scalar part is 0 and vector part is the required 3-vector. Thus it is possible to call `Quaternion.rotate(q)` with another quaternion object as an input.

**Returns:** the rotated vector returned as the same type it was specified at input.

	rotated_tuple 		= my_quaternion.rotate((1, 0, 0)) # Returns a tuple
	rotated_list  		= my_quaternion.rotate([1.0, 0.0, 0.0]) # Returns a list
	rotated_array 		= my_quaternion.rotate(numpy.array([1.0, 0.0, 0.0])) # Returns a Numpy 3-array
	rotated_quaternion	= my_quaternion.rotate(Quaternion(vector=[1, 0, 0])) # Returns a Quaternion object

**Raises:**

* `TypeError` if any of the vector elements cannot be converted to a real number.
* `ValueError` if `vector` cannot be interpreted as a 3-vector or a Quaternion object.


## Interpolation

> **`Quaternion.slerp(q0, q1, amount=0.5)`** - *class method*

Find a valid quaternion rotation at a specified distance along the minor arc of a great circle passing through any two existing quaternion endpoints lying on the unit radius hypersphere. [Source](http://en.wikipedia.org/wiki/Slerp#Quaternion_Slerp)

This is a class method and is called as a method of the class itself rather than on a particular instance.

**Params:**

* `q0` - first endpoint rotation as a Quaternion object
* `q1` - second endpoint rotation as a Quaternion object
* `amount` - interpolation parameter between 0 and 1. This describes the linear placement position of the result along the arc between endpoints; 0 being at `q0` and 1 being at `q1`. Defaults to the midpoint (0.5).

**Returns:**
a new Quaternion object representing the interpolated rotation. This is guaranteed to be a unit quaternion.

**Note:** This feature only makes sense when interpolating between unit quaternions (those lying on the unit radius hypersphere). Calling this method will implicitly normalise the endpoints to unit quaternions if they are not already unit length.

	q0 = Quaternion(axis=[1, 1, 1], angle=0.0)
	q1 = Quaternion(axis=[1, 1, 1], angle=3.141592)
	q  = Quaternion.slerp(q0, q1, 2.0/3.0) # Rotate 120 degrees (2 * pi / 3)


> **`Quaternion.intermediates(q_start, q_end, n, include_endpoints=False)`** - *class method*

Generator method to get an iterable sequence of `n` evenly spaced quaternion rotations between any two existing quaternion endpoints lying on the unit radius hypersphere. This is a convenience function that is based on `Quaternion.slerp()` as defined above.

This is a class method and is called as a method of the class itself rather than on a particular instance.

**Params:**

* `q_start` - initial endpoint rotation as a Quaternion object
* `q_end` - final endpoint rotation as a Quaternion object
* `n` - number of intermediate quaternion objects to include within the interval
* `include_endpoints` - [optional] - If set to `True`, the sequence of intermediates will be 'bookended' by `q_start` and `q_end`, resulting in a sequence length of `n + 2`. If set to `False`, endpoints are not included. Defaults to `False`.

**Yields:**
a generator object iterating over a sequence of intermediate quaternion objects.

**Note:** This feature only makes sense when interpolating between unit quaternions (those lying on the unit radius hypersphere). Calling this method will implicitly normalise the endpoints to unit quaternions if they are not already unit length.

	q0 = Quaternion(axis=[1, 1, 1], angle=0.0)
	q1 = Quaternion(axis=[1, 1, 1], angle=2 * 3.141592 / 3)
	for q in Quaternion.intermediates(q0, q1, 8, include_endpoints=True):
		v = q.rotate([1, 0, 0])
		print(v)

## Differentiation
> **`derivative(rate)`**

Get the instantaneous quaternion derivative representing a quaternion rotating at a 3D rate vector `rate`

**Params:**

* `rate` - numpy 3-array (or array-like) describing rotation rates about the global x, y and z axes respectively.

**Returns:** A unit quaternion describing the rotation rate

	q_dot = my_quaternion.derivative([0, 0, 3.14159]) # Rotate about z at 0.5 rotation per second

**Raises:**

* `TypeError`  if any of `rate` contents cannot be converted to a real number.
* `ValueError` if `rate` contains less/more than 3 elements



## Integration
> **`integrate(rate, timestep)`**

Advance a time varying quaternion to its value at a time `timestep` in the future.

The Quaternion object will be modified to its future value. It is guaranteed to remain a unit quaternion.

**Params:**

* `rate` - numpy 3-array (or array-like) describing rotation rates about the global x, y and z axes respectively.
* `timestep` - interval over which to integrate into the future. Assuming *now* is `T=0`, the integration occurs over the interval `T=0` to `T=timestep`. Smaller intervals are more accurate when `rate` changes over time.

**Note 1:** This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one. Many quaternion integration algorithms will have unwanted scaling effects leading a quaternion object to become non-unit over time, thus the object is re-normalised with each call to `integrate()`. Because this method is often called very frequently (every `timestep` for realtime simulation) an optimised re-normalisation is performed. See `_fast_normalise()` for more info.

**Note 2:** The solution is in closed form given the assumption that `rate` is constant over the interval of length `timestep`. This algorithm is not an exact solution to the differential equation over any interval where the angular rates are not constant. It is a second order approximation, meaning the integral error contains terms proportional to `timestep ** 3` and higher powers.

	>>> q = Quaternion() # null rotation
	>>> q.integrate([2*pi, 0, 0], 0.25) # Rotate about x at 1 rotation per second
	>>> q == Quaternion(axis=[1, 0, 0], angle=(pi/2))
	True
	>>>

**Raises:**

* `TypeError`  if any of `rate` contents cannot be converted to a real number.
* `ValueError` if `rate` contains less/more than 3 elements

## Accessing matrix form
> **`rotation_matrix` & `transformation_matrix`**

Get the 3x3 rotation or 4x4 homogeneous transformation matrix equivalent of the quaternion rotation.

**Returns:**

* `Quaternion.rotation_matrix` : a 3x3 orthogonal rotation matrix as a 3x3 Numpy array
* `Quaternion.transformation_matrix` : a 4x4 homogeneous transformation matrix as a 4x4 Numpy array

**Note 1:** This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.

**Note 2:** Both matrices and quaternions avoid the singularities and discontinuities involved with rotation in 3 dimensions by adding extra dimensions. This has the effect that different values could represent the same rotation, for example quaternion q and -q represent the same rotation. It is therefore possible that, when converting a rotation sequence, the output may jump between different but equivalent forms. This could cause problems where subsequent operations such as differentiation are done on this data. Programmers should be aware of this issue.

	R = my_quaternion.rotation_matrix 		  # 3x3 rotation matrix
	T = my_quaternion.transformation_matrix   # 4x4 transformation matrix

## Accessing rotation axis
> **`axis`** or **`get_axis(undefined=[0,0,0])`**

Get the axis or vector about which the quaternion rotation occurs

For a null rotation (a purely real quaternion), the rotation angle will always be `0`, but the rotation axis is undefined. It is by default assumed to be `[0, 0, 0]`.

**Note:** In the case of a null rotation, retrieving the axis is geometrically meaningless, as it could be any of an infinite set of vectors.
By default, (`[0, 0, 0]`) is returned in this instance, but should this causes undesired behaviour, please use the
alternative `get_axis()` form, specifying the `undefined` keyword to return a vector of your choice.

**Params:**

* `undefined` - [optional] - specify the axis vector that should define a null rotation. 

**Returns:** a Numpy unit 3-vector describing the Quaternion object's axis of rotation.

**Note 1:** This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.

**Note 2:** Both matrices and quaternions avoid the singularities and discontinuities involved with rotation in 3 dimensions by adding extra dimensions. This has the effect that different values could represent the same rotation, for example quaternion q and -q represent the same rotation. It is therefore possible that, when converting a rotation sequence to axis/angle representation, the output may jump between different but equivalent forms. This could cause problems where subsequent operations such as differentiation are done on this data. Programmers should be aware of this issue.

	u = my_quaternion.axis # Unit vector about which rotation occurs  #or 
	u = my_quaternion.get_axis(undefined=[1, 0, 0]) # Prefers a custom axis vector in the case of undefined result


## Accessing rotation angle
> **`angle`**, **`degrees`** or **`radians`**

Get the angle (in radians) describing the magnitude of the quaternion rotation about its rotation axis. This is guaranteed to be within the range (-pi:pi) with the direction of rotation indicated by the sign.

When a particular rotation describes a 180 degree rotation about an arbitrary axis vector `v`, the conversion to axis / angle representation may jump discontinuously between all permutations of `(-pi, pi)` and `(-v, v)`, each being geometrically equivalent (see Note 2 below).

**Returns:** a real number in the range (-pi:pi) describing the angle of rotation in radians about a Quaternion object's axis of rotation.

**Note 1:** This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.

**Note 2:** Both matrices and quaternions avoid the singularities and discontinuities involved with rotation in 3 dimensions by adding extra dimensions. This has the effect that different values could represent the same rotation, for example quaternion q and -q represent the same rotation. It is therefore possible that, when converting a rotation sequence to axis/angle representation, the output may jump between different but equivalent forms. This could cause problems where subsequent operations such as differentiation are done on this data. Programmers should be aware of this issue.


	theta = my_quaternion.angle # Magnitude of rotation about the prescribed axis, in radians
	theta = my_quaternion.radians # Equivalent, but explicit
	theta = my_quaternion.degrees # The same, but in degrees


## Accessing real components
> **`scalar` or `real`**

Get the real or scalar component of the Quaternion object

A quaternion can be described in terms of a scalar and vector part, q = [r, **v**] where:

* r is the scalar coefficient of the real part of the quaternion i.e. **a** in a + b*i* + c*j* + d*k*
* **v** is the 3-vector of coefficients to the imaginary parts of the quaternion i.e. [b, c, d] in a + b*i* + c*j* + d*k*

This property returns r

**Returns** the scalar, real valued element of the Quaternion object

	r = my_quaternion.scalar
	r = my_quaternion.real

## Accessing imaginary components
> **`vector` or `imaginary`**

Get the imaginary or vector component of the Quaternion object. This can be used, for example, to extract the stored vector when a pure-imaginary quaternion object is used to describe a vector within the three-dimensional vector space.

A quaternion can be described in terms of a scalar and vector part, q = [r, **v**] where:

* r is the scalar coefficient of the real part of the quaternion i.e. **a** in a + b*i* + c*j* + d*k*
* **v** is the 3-vector of coefficients to the imaginary parts of the quaternion i.e. [b, c, d] in a + b*i* + c*j* + d*k*

This property returns **v**

**Returns** Numpy 3-array of the 3 imaginary elements of the Quaternion object

	v = my_quaternion.vector
	v = my_quaternion.imaginary

## Accessing individual elements
> **`elements`**

Return all four elements of the quaternion object. Result is not guaranteed to be a unit 4-vector.

**Returns:** a numpy 4-array of real numbered coefficients.

	>>> a = my_quaternion.elements
	>>> print("{} + {}i + {}j + {}k".format(a[0], a[1], a[2], a[3]))
	    -0.6753741977725701 + 0.4624451782281068i + -0.059197245808339134j + 0.5714103921047806k

> **`__getitem__(index)`**

`my_quaternion[i]` returns the real numbered element at the specified index `i` in the quaternion 4-array

**Params:**

* `index` - integer in the range [-4:3] inclusive


```
>>> print("{} + {}i + {}j + {}k".format(my_quaternion[0], my_quaternion[1], my_quaternion[2], my_quaternion[3]))
   -0.6753741977725701 + 0.4624451782281068i + -0.059197245808339134j + 0.5714103921047806k
>>> print("{} + {}i + {}j + {}k".format(my_quaternion[-4], my_quaternion[-3], my_quaternion[-2], my_quaternion[-1]))
   -0.6753741977725701 + 0.4624451782281068i + -0.059197245808339134j + 0.5714103921047806k
>>>
```

**Raises:**

* `IndexError` if the index provided is invalid
* `TypeError` or `ValueError` if the index cannot be interpreted as an integer

## Modifying individual elements
> **`__setitem__(index, value)`**

`my_quaternion[i] = x` sets the element at the specified index `i` in the quaternion 4-array to the specified value `x`.

**Params:**

* `index` - integer in the range [-4:3] inclusive
* `value` - real value to be inserted into the quaternion array at `index`


```
>>> str(my_quaternion)
    '-0.653 -0.127i -0.220j +0.714k'
>>> my_quaternion[2] = 9
>>> str(my_quaternion)
    '-0.653 -0.127i +9.000j +0.714k'
>>>
```

**Raises:**

* `IndexError` if the index provided is invalid
* `TypeError` or `ValueError` if the value cannot be interpreted as a real number


# Quaternion Operations

This section defines operations applicable to pyquaternion's Quaternion objects.

The code examples below assume the existence of a Quaternion object. You can recreate this by running the following in your Python interpreter of choice:

	my_quaternion = Quaternion.random()

## String Representation
> **`__str__()`**

`str(my_quaternion)` returns an informal, nicely printable string representation of the Quaternion object. [Source](https://docs.python.org/3.4/reference/datamodel.html#object.__str__)

	>>> str(my_quaternion)
		'-0.810 +0.022i -0.563j -0.166k'
	>>> print(my_quaternion)
		-0.810 +0.022i -0.563j -0.166k
	>>>

> **`__repr__()`**

`repr(my_quaternion)` returns the 'official' string representation of the Quaternion object.  This is a string representation of a valid Python expression that could be used to recreate an object with the same value (given an appropriate environment). [Source](https://docs.python.org/3.4/reference/datamodel.html#object.__repr__)

	>>> repr(my_quaternion)
		'Quaternion(-0.80951530224438595, 0.022231097065902788, -0.56268832802625091,-0.16604999023923223)'
	>>> my_quaternion
		Quaternion(-0.80951530224438595, 0.022231097065902788, -0.56268832802625091,-0.16604999023923223)
	>>>

> **`__format__(format_spec)`**

`a_string_containing_{format_spec}_placeholders.format(my_quaternion)` inserts a customisable, nicely printable string representation of the Quaternion object into the respective places in the provided string. [Source](https://docs.python.org/3.4/reference/datamodel.html#object.__format__)

The syntax for `format_spec` mirrors that of the built in format specifiers for floating point types. Check out the official Python [format specification mini-language](https://docs.python.org/3.4/library/string.html#formatspec) for details.
An empty `format_spec` string will result in the same behaviour as the `Quaternion.__str__()`.

	>>> "My quaternion is: {}".format(my_quaternion)
		'My quaternion is: -0.810 +0.022i -0.563j -0.166k'
	>>> "My quaternion is: {:+.6}".format(my_quaternion)
		'My quaternion is: -0.809515 +0.0222311i -0.562688j -0.16605k'


## Bool
> **`__bool__()` or `__nonzero__()`**

**Returns:** `False` within a logical context if the Quaternion object is zero, i.e. `Quaternion(0.0, 0.0, 0.0, 0.0)` or `True` otherwise.

The bitwise not operator `~` can be used to invert the boolean value, however the keyword `not` (logical) is preferred.

**Note:** This does not evaluate the booleanity of a quaternion rotation. A non-zero Quaternion object such as `Quaternion(1.0, 0.0, 0.0, 0.0)` will have a boolean value of `True` even though it represents a **null** rotation.

	>>> Quaternion() == True
	True
	>>> not Quaternion() == False
	True
	>>> Quaternion(scalar=0.0) == False
	True


## Equality
> **`__eq__(other)`**

`q1 == q2` returns `True` if all corresponding elements are equal between two Quaternion objects `q1` and `q2`, or `False` otherwise.

The inequality operator `!=` can also be used to verify inequality in a similar way.

Because comparisons are carried out on floating point elements, equality is considered `True` when the absolute difference between elements falls below a threshold error. This is determined by [numpy.allclose()](http://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html) with an absolute tolerance of `1.0e-14` and a relative tolerance of `1.0e-13`. As a result, objects differing by very small individual element differences may be considered equal.

**Note:** This does not directly evaluate the equality of a quaternion rotation. For example, unit Quaternions q and -q will have an equality of `False` even though they represent the equivalent rotation.

	>>> Quaternion(1, 0, 1, 1) == Quaternion(scalar=1.0, vector=[0.0, 1.0, 1.0])
	True
	>>> Quaternion(1, 0, 1, 1) == Quaternion(scalar=1.0, vector=[0.1, 1.0, 1.0])
	False
	>>> Quaternion() != Quaternion(scalar=2)
	True


## Negation
> **`__neg__()`**

`-q` is the quaternion formed by the element wise negation of the elements of `q`.

**Returns:** a new Quaternion object representing the negation of the single operand.
If the operand is a unit quaternion, the result is guaranteed to be a unit quaternion.

	>>> my_elements = my_quaternion.elements() # Numpy array of individual elements
	>>> -my_quaternion == Quaternion(-my_elements)
	True


## Addition
> **`__add__(other)`**

`q1 + q2` is the quaternion formed by element-wise sum of `q1` and `q2`. [Source][arithmetic]

**Note:** If 'other' is not a Quaternion object, it will be converted to one, using the behaviour described in the [object initialisation][initialisation] section. As described therein, a `TypeError` or `ValueError` will be raised if this conversion fails.

**Returns:** a new Quaternion object representing the sum of the inputs.
The sum is **not** guaranteed to be a unit quaternion.

	>>> q1 = Quaternion.random()
	>>> q2 = Quaternion.random()
	>>> q1 + q2 == Quaternion(q1.elements() + q2.elements())
	True

## Subtraction
> **`__sub__(other)`**

`q1 - q2` is the quaternion formed by element-wise difference between `q1` and `q2`. [Source][arithmetic]

**Note:** If 'other' is not a Quaternion object, it will be converted to one, using the behaviour described in the [object initialisation][initialisation] section. As described therein, a `TypeError` or `ValueError` will be raised if this conversion fails.

**Returns:** a new Quaternion object representing the difference of the inputs.
The difference is **not** guaranteed to be a unit quaternion.

	>>> q1 = Quaternion.random()
	>>> q2 = Quaternion.random()
	>>> q1 - q2 == Quaternion(q1.elements() - q2.elements())
	True


## Multiplication
> **`__mul__(other)`**

`q1 * q2` is the quaternion formed by Hamilton product of `q1` and `q2`. [Source][arithmetic]

The Hamiltonian product is not commutative. Ensure your operands are correctly placed.

**Note:** If 'other' is not a Quaternion object, it will be converted to one, using the behaviour described in the [object initialisation][initialisation] section. As described therein, a `TypeError` or `ValueError` will be raised if this conversion fails. As a result this operation holds true for scalar multiplication as scalars are converted to pure real Quaternion objects.

**Returns:** a new Quaternion object representing the Hamilton product of the inputs.
If the two multiplicands are unit quaternions, the product is guaranteed to be a unit quaternion.

	>>> one = Quaternion(1, 0, 0, 0)
	>>> i   = Quaternion(0, 1, 0, 0)
	>>> j   = Quaternion(0, 0, 1, 0)
	>>> k   = Quaternion(0, 0, 0, 1)
	>>> (i * i) == (j * j) == (k * k) == (i * j * k) == -1
	True


## Division
> **`__truediv__(other)` or `__div__(other)`**

`q1 / q2` is the quaternion formed by Hamilton product of `q1` and `q2.inverse()`. [Source][arithmetic]

The Hamiltonian product is not commutative. Ensure your operands are correctly placed.



**Note:** If 'other' is not a Quaternion object, it will be converted to one, using the behaviour described in the [object initialisation][initialisation] section. As described therein, a `TypeError` or `ValueError` will be raised if this conversion fails. As a result this operation holds true for scalar division as scalars are converted to pure real Quaternion objects.

**Returns:** a new Quaternion object representing the Hamilton quotient of the inputs.
If the dividend and divisor are unit quaternions, the quotient is guaranteed to be a unit quaternion.

	>>> my_quaternion / my_quaternion == Quaternion(1.0)
	True


## Exponentiation
> **`__pow__(other)`**

`q ** p` is the quaternion formed by raising the Quaternion `q1` to the power of `p` for any real `p`. [Source](http://en.wikipedia.org/wiki/Quaternion#Exponential.2C_logarithm.2C_and_power)

**Returns:** a new Quaternion object representing the the object raised to the power of the input.
If the base object is a unit quaternion, the result is guaranteed to be a unit quaternion.

	>>> one = Quaternion(1, 0, 0, 0)
	>>> i   = Quaternion(0, 1, 0, 0)
	>>> j   = Quaternion(0, 0, 1, 0)
	>>> k   = Quaternion(0, 0, 0, 1)
	>>> (i ** 2) == (j ** 2) == (k ** 2) == -1
	True

**Raises:** `TypeError` if `other` cannot be interpreted as a real number.

[arithmetic]: http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/arithmetic/index.htm
