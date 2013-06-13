import numpy as np
import unittest
import space
from data import Data

class TestData(unittest.TestCase):
    """ Test the Data class. """

    def setUp(self):
        self.valid_dtypes = (np.float32, np.float64, np.complex64, np.complex128)

    def test_get_dtype(self):
        """ Test the _get_dtype function. """
        d = Data()
        self.assertRaises(TypeError, d._get_dtype, np.int)
        self.assertRaises(TypeError, d._get_dtype, 'abc')
        for dtype in self.valid_dtypes:
            d._get_dtype(dtype)
    
    def test_set_gce_type(self):
        """ Test the _set_gce_type function. """
        d = Data()
        d._set_gce_type('grid')
        d._set_gce_type('const')
        d._set_gce_type('out')
        self.assertRaises(TypeError, d._set_gce_type, 'other')


    def test_to_and_from_gpu(self):
        """ Make sure we can load and unload data off the gpu. """
        shape = (100,100,100)
        d = Data()
        space.initialize_space(shape)

        for dtype in self.valid_dtypes:
            # Create data to load.
            d_cpu = np.random.randn(*shape).astype(dtype)
            if dtype in (np.complex64, np.complex128):
                d_cpu = (1 + 1j) * d_cpu

            # Load and retrieve.
            d.to_gpu(d_cpu)
            self.assertTrue((d_cpu == d.get()).all())


if __name__ == '__main__':
    unittest.main()

