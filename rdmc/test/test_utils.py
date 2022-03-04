#!/usr/bin/env python3

"""
Unit tests for the utils module.
"""

import logging
import unittest

import numpy as np
from rdmc.utils import reverse_map

logging.basicConfig(level=logging.DEBUG)

################################################################################

class TestUtils(unittest.TestCase):
    """
    The general class to test functions in the utils module
    """

    def test_reverse_match(self):
        """
        Test the functionality to reverse a mapping.
        """
        map = [ 1,  2,  3,  4,  5, 17, 18, 19, 20, 21, 22, 23, 24, 25,  6,  7,  8,
                9, 10, 11, 12, 13, 14, 15, 16, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                35, 36, 37, 38, 39]
        r_map = [ 0,  1,  2,  3,  4, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,  5,
                  6,  7,  8,  9, 10, 11, 12, 13, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                  34, 35, 36, 37, 38]

        self.assertSequenceEqual(r_map, reverse_map(map))
        np.testing.assert_equal(np.array(r_map), reverse_map(map, as_list=False))


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=3))
