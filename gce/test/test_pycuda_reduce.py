import pycuda.autoinit
from pycuda import gpuarray
import numpy
import unittest

class TestPycudaReduce(unittest.TestCase):
    def setUp(self):
        self.shapes = [(10000,), (100000,), (1000000,), (10000000,), \
            (100,100), (1000,1000), \
            (10,20,30), (40,50,60), (200,200,200)]

    def test_dot(self):
        """ Test dot-product. """
        dtypes = [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128]
        for dtype in dtypes:
            for shape in self.shapes:
                x = gpuarray.to_gpu(numpy.random.randn(*shape).astype(dtype))
                y = gpuarray.to_gpu(numpy.random.randn(*shape).astype(dtype))

                dot_cpu = numpy.dot(x.get().flatten(), y.get().flatten()) 
                dot_gpu = gpuarray.dot(x, y).get()

                percent_error = abs(dot_cpu-dot_gpu)/abs(dot_cpu)*100
#                 print 'shape:', shape
#                 print 'data type:', dtype 
#                 print 'numpy computed dot product:', dot_cpu
#                 print 'gpuarray computed dot product:', dot_gpu
#                 print 'percent error:', percent_error, '%'
#                 print '\n'

                self.assertTrue(percent_error < 10.0, 'Error above 10%.')

    def test_sum(self):
        """ Test sum. """
        dtypes = [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128]
        for dtype in dtypes:
            for shape in self.shapes:
                x = gpuarray.to_gpu(numpy.random.randn(*shape).astype(dtype))

                res_cpu = numpy.sum(x.get().flatten())
                res_gpu = gpuarray.sum(x).get()

                percent_error = abs(res_cpu-res_gpu)/abs(res_cpu)*100
#                 print 'shape:', shape
#                 print 'data type:', dtype 
#                 print 'numpy computed result:', res_cpu
#                 print 'gpuarray computed result:', res_gpu
#                 print 'percent error:', percent_error, '%'
#                 print '\n'

                self.assertTrue(percent_error < 10.0, 'Error above 10%.')

if __name__ == '__main__':
    unittest.main()
