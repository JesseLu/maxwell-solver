import numpy as np
import unittest
from grid import Grid 
from const import Const
from out import Out
import space
from kernel import Kernel

class TestExample(unittest.TestCase):
    """ An example of how to use GCE. 
    
    Execute using 'python test_example.py' from command line.

    """

    def test_simple_example(self):
        """ Implement a simple kernel. """
        # Form data to work on.
        shape = (100,100,100)
        space.initialize_space(shape)
        x = Grid((1 + 1j) * np.ones(shape).astype(np.complex128))
        z = Out(np.float64)

        # Make a kernel.
        code = """  
                if (_in_global) { // Need to be in the space.
                    z += real(x(0,0,0)) + imag(x(0,0,0));
                } """
        fun = Kernel(code, \
                    ('x', 'grid', x.dtype), \
                    ('z', 'out', z.dtype))

        # Execute and check the result.
        fun(x, z)
        gpu_sum = z.get()
        # print 'Answer:', gpu_sum, '(should be 2e6).'

if __name__ == '__main__':
    unittest.main()
