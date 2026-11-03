import numpy as np
import pytest

from pyquaternion import Quaternion


@pytest.mark.parametrize("index", range(4))
def test_copy_constructor_has_independent_elements(index):
    original = Quaternion(1, 2, 3, 4)
    copied = Quaternion(original)
    np.testing.assert_array_equal(copied.elements, original.elements)
    assert not np.shares_memory(original.elements, copied.elements)
    copied[index] = 99
    np.testing.assert_array_equal(original.elements, [1, 2, 3, 4])
    original[(index + 1) % 4] = -99
    assert copied[(index + 1) % 4] != -99


def test_unit_copy_does_not_alias_original():
    original = Quaternion(axis=[0, 0, 1], angle=0.7)
    before = original.elements.copy()
    normalized = original.normalised
    normalized[0] = 0
    np.testing.assert_array_equal(original.elements, before)
