# pyquaternion

[![Build Status](https://travis-ci.org/KieranWynn/pyquaternion.svg?branch=master)](https://travis-ci.org/KieranWynn/pyquaternion)

A fully featured, pythonic library for quaternion representation, manipulation, 3D animation and geometry.

Please visit the **[pyquaternion homepage](http://kieranwynn.github.io/pyquaternion/)** for full information and the latest documentation.

**Designed for Python 2.7+ and 3.0+**

![demo](./demo.gif)

> Example: Smooth animation with interpolation between random orientations using the pyquaternion module.

## Quickstart

Install from [PyPI](https://pypi.python.org/pypi/pyquaternion/0.9.0)

```$ pip install pyquaternion```

Run the following for a basic overview. A copy of this example can be found in [demo.py](./demo/demo.py).

```python
import pyquaternion


# Create a quaternion representing a rotation of +90 degrees about positive y axis.
my_quaternion = pyquaternion.Quaternion(axis=[0, 1, 0], degrees=90)

my_vector = [0, 0, 4]
my_rotated_vector = my_quaternion.rotate(my_vector)

print('\nBasic Rotation')
print('--------------')
print('My Vector: {}'.format(my_vector))
print('Performing rotation of {angle} deg about {axis}'.format(angle=my_quaternion.degrees, axis=my_quaternion.axis))
print('My Rotated Vector: {}'.format(my_rotated_vector))


# Create another quaternion representing no rotation at all
null_quaternion = pyquaternion.Quaternion(axis=[0, 1, 0], angle=0)

print('\nInterpolated Rotation')
print('---------------------')

# The following will create a sequence of 9 intermediate quaternion rotation objects
for q in pyquaternion.Quaternion.intermediates(null_quaternion, my_quaternion, 9, include_endpoints=True):
    my_interpolated_point = q.rotate(my_vector)
    print('My Interpolated Point: {point}\t(after rotation of {angle} deg about {axis})'.format(
        point=my_interpolated_point, angle=round(q.degrees, 4), axis=q.axis
    ))
    
print('Done!')
````

Example output:

```
Basic Rotation
--------------
My Vector: [0, 0, 4]
Performing rotation of 90.0 deg about [ 0.  1.  0.]
My Rotated Vector: [4.0, 0.0, 0.0]

Interpolated Rotation
---------------------
My Interpolated Point: [0.0, 0.0, 4.0]	(after rotation of 0.0 deg about [ 0.  0.  0.])
My Interpolated Point: [0.62573786016092348, 0.0, 3.9507533623805511]	(after rotation of 9.0 deg about [ 0.  1.  0.])
My Interpolated Point: [1.2360679774997898, 0.0, 3.8042260651806146]	(after rotation of 18.0 deg about [ 0.  1.  0.])
My Interpolated Point: [1.8159619989581872, 0.0, 3.5640260967534712]	(after rotation of 27.0 deg about [ 0.  1.  0.])
My Interpolated Point: [2.3511410091698921, 0.0, 3.2360679774997894]	(after rotation of 36.0 deg about [ 0.  1.  0.])
My Interpolated Point: [2.8284271247461903, 0.0, 2.8284271247461898]	(after rotation of 45.0 deg about [ 0.  1.  0.])
My Interpolated Point: [3.2360679774997894, 0.0, 2.3511410091698921]	(after rotation of 54.0 deg about [ 0.  1.  0.])
My Interpolated Point: [3.5640260967534712, 0.0, 1.8159619989581879]	(after rotation of 63.0 deg about [ 0.  1.  0.])
My Interpolated Point: [3.8042260651806146, 0.0, 1.2360679774997898]	(after rotation of 72.0 deg about [ 0.  1.  0.])
My Interpolated Point: [3.9507533623805515, 0.0, 0.62573786016092403]	(after rotation of 81.0 deg about [ 0.  1.  0.])
My Interpolated Point: [4.0, 0.0, 0.0]	(after rotation of 90.0 deg about [ 0.  1.  0.])
Done!
````

