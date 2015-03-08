#Quaternion Operations
This page defines operations applicable to pyquaternion's Quaternion objects.

The code examples below assume the existence of a Quaternion object. You can recreate this by running the following in your Python interpreter of choice:
	
	my_quaternion = Quaternion.random()
	
## String Representation
> **`__str__()`**

An informal, nicely printable string representation of the Quaternion object. [Source](https://docs.python.org/3.4/reference/datamodel.html#object.__str__)

	>>> str(my_quaternion)
		'-0.810 +0.022i -0.563j -0.166k'

___

> **`__repr__()`**

The 'official' string representation of the Quaternion object.  This is a string representation of a valid Python expression that could be used to recreate an object with the same value (given an appropriate environment). [Source](https://docs.python.org/3.4/reference/datamodel.html#object.__repr__)

	>>> repr(my_quaternion)
		'Quaternion(-0.80951530224438595, 0.022231097065902788, -0.56268832802625091,-0.16604999023923223)'

___
		
> **`__format__(format_spec)`**

A customisable, nicely printable string representation of the Quaternion object. [Source](https://docs.python.org/3.4/reference/datamodel.html#object.__format__)

The syntax for `format_spec` mirrors that of the built in format specifiers for floating point types. Check out the official Python [format specification mini-language](https://docs.python.org/3.4/library/string.html#formatspec) for details.
An empty `format_spec` string will result in the same behaviour as the `Quaternion.__str__()`.

	>>> "My quaternion is: {}".format(my_quaternion)
		'My quaternion is: -0.810 +0.022i -0.563j -0.166k'
	>>> "My quaternion is: {:+.6}".format(my_quaternion)
		'My quaternion is: -0.809515 +0.0222311i -0.562688j -0.16605k'
	
	
## Bool
> **`__bool__()` or `__nonzero__()`**

**Returns:** `False` if the Quaternion object is zero, i.e. `Quaternion(0.0, 0.0, 0.0, 0.0)` or `True` otherwise. 

The bitwise not operator `~` can be used to invert the boolean value, however the keyword `not` (logical) is preferred.

> **Note:** This does not evaluate the booleanity of a quaternion rotation. A non-zero Quaternion object such as `Quaternion(1.0, 0.0, 0.0, 0.0)` will have a boolean value of `True` even though it represents a **null** rotation.

	>>> Quaternion() == True
	True
	>>> not Quaternion() == False
	True
	>>> Quaternion(scalar=0.0) == False
	True


## Equality
> **`__eq__(other)`**

**Returns:** `True` if all corresponding elements are equal between two Quaternion objects, or `False` otherwise. 

The inequality operator `!=` can also be used to verify inequality in a similar way.

Because comparisons are carried out on floating point elements, equality is considered `True` when the absolute difference between elements falls below a threshold error. This is determined by [numpy.allclose()](http://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html) with an absolute tolerance of `1.0e-14` and a relative tolerance of `1.0e-13`. As a result, objects differing by very small individual element differences may be considered equal.

> **Note:** This does not directly evaluate the equality of a quaternion rotation. For example, unit Quaternions q and -q will have an equality of `False` even though they represent the equivalent rotation.

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

> **Note:** If 'other' is not a Quaternion object, it will be converted to one, using the behaviour described in the [object initialisation][initialisation] section. As described therein, a `TypeError` or `ValueError` will be raised if this conversion fails.

**Returns:** a new Quaternion object representing the sum of the inputs.
The sum is **not** guaranteed to be a unit quaternion.

	>>> q1 = Quaternion.random()
	>>> q2 = Quaternion.random()
	>>> q1 + q2 == Quaternion(q1.elements() + q2.elements())
	True

## Subtraction
> **`__sub__(other)`**

`q1 - q2` is the quaternion formed by element-wise difference between `q1` and `q2`. [Source][arithmetic] 

> **Note:** If 'other' is not a Quaternion object, it will be converted to one, using the behaviour described in the [object initialisation][initialisation] section. As described therein, a `TypeError` or `ValueError` will be raised if this conversion fails.

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

> **Note:** If 'other' is not a Quaternion object, it will be converted to one, using the behaviour described in the [object initialisation][initialisation] section. As described therein, a `TypeError` or `ValueError` will be raised if this conversion fails. As a result this operation holds true for scalar multiplication as scalars are converted to pure real Quaternion objects.

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



> **Note:** If 'other' is not a Quaternion object, it will be converted to one, using the behaviour described in the [object initialisation][initialisation] section. As described therein, a `TypeError` or `ValueError` will be raised if this conversion fails. As a result this operation holds true for scalar division as scalars are converted to pure real Quaternion objects.

**Returns:** a new Quaternion object representing the Hamilton quotient of the inputs. 
If the dividend and divisor are unit quaternions, the quotient is guaranteed to be a unit quaternion.

	>>> my_quaternion / my_quaternion == Quaternion(1.0)
	True


## Exponentiation
> **`__pow__(other)`**

`q ** p` is the quaternion formed by raising the Quaternion `q1` to the power of `p` for any real `p`. [Source](http://en.wikipedia.org/wiki/Quaternion#Exponential.2C_logarithm.2C_and_power)

> Raises `TypeError` if `other` cannot be interpreted as a real number.

**Returns:** a new Quaternion object representing the the object raised to the power of the input.
If the base object is a unit quaternion, the result is guaranteed to be a unit quaternion.

	>>> one = Quaternion(1, 0, 0, 0)
	>>> i   = Quaternion(0, 1, 0, 0)
	>>> j   = Quaternion(0, 0, 1, 0)
	>>> k   = Quaternion(0, 0, 0, 1)
	>>> (i ** 2) == (j ** 2) == (k ** 2) == -1
	True


[initialisation]: ./initialisation.md
[arithmetic]: http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/arithmetic/index.htm
