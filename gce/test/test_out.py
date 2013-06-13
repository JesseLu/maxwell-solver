import numpy as np
import unittest
import space
from out import Out, batch_reduce
from mpi4py.MPI import COMM_WORLD as comm
import make_test_cases 

class TestOut(unittest.TestCase):
    """ Test the Out class. """

    def setUp(self):
        """ Spaces of various sizes and dtypes to test. """
#         shapes = [(10,20,30), (40,50,60), (100,100,100), (20,100,10000)]
#         dtypes = [np.float32, np.float64, np.complex64, np.complex128] 
#         self.cases = [{'shape':s, 'dtype':t} for s in shapes for t in dtypes]
        self.cases = make_test_cases.cases

    def test_init(self):
        """ Test initialize function. """
        for case in self.cases:
            untype_array = np.zeros(case['shape']).astype(np.int)
            space.initialize_space(case['shape'])
            self.assertRaises(TypeError, Out, np.int)
            self.assertRaises(TypeError, Out, untype_array)
            self.assertRaises(TypeError, Out, 'string')
            self.assertRaises(TypeError, Out, np.complex128, op='bad')
            Out(case['dtype'])
            Out(case['dtype'], op='sum')

    def test_sum(self):
        """ Make sure summing works. """
        for case in self.cases:
            space.initialize_space(case['shape'])
            x = Out(case['dtype'], op='sum')
            x_cpu_data = np.random.randn(*case['shape'][1:]).astype(case['dtype'])
            if case['dtype'] in (np.complex64, np.complex128):
                x_cpu_data = (1 + 1j) * x_cpu_data

            x.data.set(x_cpu_data)
            res_gold = comm.allreduce(np.sum(x_cpu_data.flatten()))

            x.reduce()
            err = abs(res_gold - x.get()) / abs(res_gold)

            if case['dtype'] in (np.float32, np.complex64):
                self.assertTrue(err < 1e-3)
            else:
                self.assertTrue(err < 1e-10)


    def test_batch_sum(self):
        """ Make sure batch summing works. """
        num_outs = 3
        for case in self.cases:
            space.initialize_space(case['shape'])
            x = [Out(case['dtype'], op='sum') for k in range(num_outs)]
            x_cpu_data = [np.random.randn(*case['shape'][1:])\
                            .astype(case['dtype']) for k in range(num_outs)]
                    
            if case['dtype'] in (np.complex64, np.complex128):
                for k in range(num_outs):
                    x_cpu_data[k] = (1 + 1j) * x_cpu_data[k]

            res_gold = []
            for k in range(num_outs):
                x[k].data.set(x_cpu_data[k])
                res_gold.append(comm.allreduce(np.sum(x_cpu_data[k].flatten())))

            batch_reduce(*x)
            res_gpu = [x_indiv.get() for x_indiv in x]

            for k in range(num_outs):
                err = abs(res_gold[k] - res_gpu[k]) / abs(res_gold[k])

                if case['dtype'] in (np.float32, np.complex64):
                    self.assertTrue(err < 1e-3)
                else:
                    self.assertTrue(err < 1e-10)


if __name__ == '__main__':
    unittest.main()

