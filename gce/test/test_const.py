import numpy as np
import unittest
import space
from const import Const 
import make_test_cases 

class TestConst(unittest.TestCase):
    """ Test the Const class. """

    def setUp(self):
        """ Spaces of various sizes and dtypes to test. """
#         shapes = [(10,20,30), (40,50,60), (100,100,100)]
#         dtypes = [np.float32, np.float64, np.complex64, np.complex128] 
#         self.cases = [{'shape':s, 'dtype':t} for s in shapes for t in dtypes]
        self.cases = make_test_cases.cases

    def test_init(self):
        """ Test initialize function. """
        for case in self.cases:
            untype_array = np.zeros(case['shape']).astype(np.int)
            space.initialize_space(case['shape'])
            self.assertRaises(TypeError, Const, np.int)
            self.assertRaises(TypeError, Const, untype_array)
            self.assertRaises(TypeError, Const, 'string')
            Const(np.random.randn(10).astype(case['dtype']))


if __name__ == '__main__':
    unittest.main()

