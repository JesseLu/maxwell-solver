import numpy as np
import unittest
from grid import Grid 
from const import Const
from out import Out
import space
from kernel import Kernel
from jinja2 import Template
from mpi4py.MPI import COMM_WORLD as comm
import make_test_cases 

class TestKernel(unittest.TestCase):
    """ Test the Kernel class. """

    def setUp(self):
        """ Spaces of various sizes and dtypes to test. """
#         shapes = [(41,23,31), (48,59,67), (100,100,100)]
#         dtypes = [np.float32, np.float64, np.complex64, np.complex128] 
#         shapes = [shapes[2]]
#         dtypes = [np.float32, np.complex128]
#         self.cases = [{'shape':s, 'dtype':t} for s in shapes for t in dtypes]
        self.cases = make_test_cases.cases

    def test_init(self):
        """ Just make sure we can initialize the kernel. """
        for case in self.cases:
            # Form data to work on.
            space.initialize_space(case['shape'])
            x_np = np.random.randn(*case['shape']).astype(case['dtype'])
            x = Grid(x_np)
            fun = Kernel('', ('x', 'grid', x.dtype))
            fun = Kernel('', ('x', 'grid', x.dtype), shape_filter='all')
            fun = Kernel('', ('x', 'grid', x.dtype), shape_filter='skinny')
            fun = Kernel('', ('x', 'grid', x.dtype), shape_filter='square')
            self.assertRaises(TypeError, Kernel, '', ('x', 'grid', x.dtype), \
                                            shape_filter1='all')
            self.assertRaises(TypeError, Kernel, '', ('x', 'grid', x.dtype), \
                                            shape_filter='blah')

    def test_grid_shape(self):
        """ Make sure the grid shapes of the exec configs are correct. """
        tot_threads = lambda gs, bs: (gs[0] * bs[0], gs[1] * bs[1])
        for case in self.cases:
            space.initialize_space(case['shape'])
            z = Out(case['dtype'])
            fun = Kernel('', ('z', 'out', z.dtype)) 
            for cfg in fun.exec_configs:
                for ind in range(2):
                    self.assertTrue(cfg['grid_shape'][ind] * \
                                    cfg['block_shape'][ind] >= \
                                    case['shape'][ind+1])
                    self.assertTrue((cfg['grid_shape'][ind]-1) * \
                                    cfg['block_shape'][ind] < \
                                    case['shape'][ind+1])
            # One padded case.
            fun = Kernel('', ('z', 'out', z.dtype), padding=(1,2,3,4)) 
            pad = [3, 7]
            for cfg in fun.exec_configs:
                for ind in range(2):
                    self.assertTrue(cfg['grid_shape'][ind] * \
                                    (cfg['block_shape'][ind]-pad[ind]) >= \
                                    case['shape'][ind+1])
                    self.assertTrue((cfg['grid_shape'][ind]-1) * \
                                    (cfg['block_shape'][ind]-pad[ind]) < \
                                    case['shape'][ind+1])

    def test_kernel_self_opt(self):
        """ Make sure the kernel settles on the fastest configuration. """
        for case in (self.cases[0],):
            space.initialize_space(case['shape'])
            z = Out(case['dtype'])
            fun = Kernel('', ('z', 'out', z.dtype), shape_filter='square') 

            # Run through all configurations.
            hist = []
            while fun.exec_configs:
                hist.append(fun(z))

            # Find fastest config, early-bird wins ties.
            best_time, best_cfg = min(hist, key=lambda x: x[0]) 

            time, next_cfg = fun(z) # Run once more, should use fastest configuration.

            # Make sure we have chosen the fastest configuration.
            self.assertEqual(best_cfg, next_cfg)


    def test_simple_kernel(self):
        """ Implement a simple kernel. """
        for case in self.cases:
            # Form data to work on.
            space.initialize_space(case['shape'])
            x_np = comm.allreduce(np.random.randn(*case['shape']).astype(case['dtype']))
            x = Grid(x_np, x_overlap=2)
            s_np = comm.allreduce(np.random.randn(case['shape'][0],1,1).astype(case['dtype']))
            s = Const(s_np)
            z = Out(case['dtype'])

            # Make a kernel.
            code = Template("""
                            if (_in_local && _in_global) {
                                z += a * s(_X) * x(0,0,0);
                                // z += a * x(0,0,0);
                            }
                            """).render()
            fun = Kernel(code, \
                        ('a', 'number', case['dtype']), \
                        ('x', 'grid', x.dtype), \
                        ('s', 'const', s.dtype), \
                        ('z', 'out', z.dtype), \
                        shape_filter='all')

            # Execute and check the result.
            # fun()
            while fun.exec_configs:
            # for k in range(40):
                fun(case['dtype'](2.0), x, s, z)
                # fun(case['dtype'](2.0), x, z)
                gpu_sum = z.get()
                cpu_sum = np.sum(2 * s_np * x_np)
                # cpu_sum = np.sum(2 * x_np)
                err = abs(gpu_sum - cpu_sum) / abs(cpu_sum)
                if case['dtype'] in (np.float32, np.complex64):
                    self.assertTrue(err < 1e-2, (case, err))
                else:
                    self.assertTrue(err < 1e-6, (case, err))

    def test_padded_kernel(self):
        """ Implement a simple padded kernel. """
        for case in self.cases:
            # Form data to work on.
            space.initialize_space(case['shape'])
            x_np = comm.allreduce(np.random.randn(*case['shape']).astype(case['dtype']))
            x = Grid(x_np, x_overlap=1)
            s_np = comm.allreduce(np.random.randn(1).astype(case['dtype']))
            s = Const(s_np)
            z = Out(case['dtype'])

            # Make a kernel.
            code = Template("""
                            if (_in_local && _in_global) {
                                x(0,0,0) = s(0) * x(0,0,0);
                                z += a * x(0,0,0);
                            }
                            """).render()
            fun = Kernel(code, \
                        ('a', 'number', case['dtype']), \
                        ('x', 'grid', x.dtype), \
                        ('s', 'const', s.dtype, s.data.size), \
                        ('z', 'out', z.dtype), \
                        padding=(1,1,1,1))

            # Execute and check the result.
            fun(case['dtype'](2), x, s, z)
            gpu_sum = z.get()
            cpu_sum = np.sum(2.0 * s_np * x_np)
            err = abs(gpu_sum - cpu_sum) / abs(cpu_sum)
            # print case, err
            if case['dtype'] in (np.float32, np.complex64):
                self.assertTrue(err < 1e-2, (case, err))
            else:
                self.assertTrue(err < 1e-6, (case, err))


if __name__ == '__main__':
    unittest.main()
