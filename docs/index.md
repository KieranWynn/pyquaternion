# Welcome
Welcome! pyquaternion is a full-featured Python module for representing and using quaternions.

The following should get you up and running in no time with pyquaternion.

<a name="dependencies"></a>
# Dependencies
pyQuaternion has been tested to work correctly on Python 2.7x and Python 3x. Compatibility outside of these versions is not guaranteed. If you really need support for other versions, feel free to fork the project on [Github](https://github.com/KieranWynn/pyquaternion) and make the necessary modifications.

pyQuaternion requires [Numpy](http://www.numpy.org) for the representation of arrays and matrices. Chances are if you're needing quaternions, you've been dealing with numerical computation already and you have it installed. If so, great, ignore the rest of this section. If not, read on!

Numpy is a core element of the [SciPy](http://www.scipy.org/about.html) package, which also contains a number of useful scientific computing and visualisation tools. If you will be using these (e.g. plotting 3D vectors with [Matplotlib](http://matplotlib.org)), you might want to look into [installing the whole SciPy stack](http://www.scipy.org/install.html), however, SciPy is not required in its entirety for pyquaternion. If you just want the important bit, proceed as follows:

Numpy can easily be installed as a standalone with python's package manager, pip. Pip comes included with recent versions of Python (2.7.9+ and 3.4+), but if you need it, follow the instructions [here](https://pip.pypa.io/en/latest/installing.html#install-pip). 
Once you have pip installed, simply do:

	$ pip install numpy

> Note: If you're using Python3, your pip command may be 

	$ pip3 install numpy


<a name="getting_started"></a>
# Getting started
The following aims to familiarize you with the basic functionality of quaternions in pyquaternion. It provides an entry point and a quick orientation (no pun intended) for those who want get stuck straight in. More comprehensive feature summaries can be found in the [features][features] and [operations][operations] documentation.

 If you want to learn more about quaternions and how they apply to certain problems, check out the [quaternion basics][quaternion basics] page.

To start, you will need to import the *Quaternion* object from the *pyquaternion* module:
	
	>>> from pyquaternion import Quaternion

> Note: if this gives you an error, you may be missing a dependency. See the [dependencies](#dependencies) section for how to install them.
	
Next, create a Quaternion object to describe your desired rotation:

	>>> my_quaternion = Quaternion(axis=[1, 0, 0], angle=3.14159265)
> Note: There are many ways to create a Quaternion object. See the [initialisation][initialisation] section for a complete guide.

Now you can do a bunch of useful things with your new quaternion object. Let's try rotating a vector:
	
	>>> import numpy
	>>> numpy.set_printoptions(suppress=True) # Suppress insignificant values for clarity
	>>> v = numpy.array([0., 0., 1.]) # Unit vector in the +z drection
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


[quaternion basics]: ./intro_to_quaternions.md
[initialisation]: ./initialisation.md
[features]: ./pyquaternion_features.md
[operations]: ./pyquaternion_operations.md