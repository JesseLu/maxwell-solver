import numpy as np
import unittest
import space
from grid import Grid
from pycuda import driver as drv
from mpi4py.MPI import COMM_WORLD as comm
import make_test_cases 

class TestGrid(unittest.TestCase):
    """ Test the Grid class. """

    def setUp(self):
        """ Spaces of various sizes and dtypes to test. """
#         shapes = [(100,20,30), (40,50,60), (100,100,100)]
#         dtypes = [np.float32, np.float64, np.complex64, np.complex128] 
#         shapes = [(100,100,100)]
#         dtypes = [np.complex128] 
#         self.cases = [{'shape':s, 'dtype':t} for s in shapes for t in dtypes]
        self.cases = make_test_cases.cases

    def test_init(self):
        """ Test initialize function. """
        for case in self.cases:
            unfit_array = np.zeros(10)
            untype_array = np.zeros(case['shape']).astype(np.int)
            space.initialize_space(case['shape'])
            self.assertRaises(TypeError, Grid, np.int)
            self.assertRaises(TypeError, Grid, unfit_array)
            self.assertRaises(TypeError, Grid, untype_array)
            self.assertRaises(TypeError, Grid, 'string')
            self.assertRaises(TypeError, Grid, np.float32, x_overlap='a')
            self.assertRaises(TypeError, Grid, np.float32, x_overlap=2.2)
            self.assertRaises(TypeError, Grid, np.float32, x_overlap=-2)
            Grid(np.random.randn(*case['shape']).astype(case['dtype']))
            Grid(case['dtype'])
            Grid(np.random.randn(*case['shape']).astype(case['dtype']), x_overlap=1)
            Grid(case['dtype'], x_overlap=2)

    def test_recover(self):
        """ Make sure we can store and retrieve information from the GPU. """
        for case in self.cases:
            space.initialize_space(case['shape'])
            data = np.random.randn(*case['shape']).astype(case['dtype'])
            cpu_data = np.empty_like(data)
            comm.Allreduce(data, cpu_data)
            g = Grid(cpu_data)
            gpu_data = g.get()
            if comm.Get_rank() == 0:
                self.assertTrue((cpu_data == gpu_data).all())

            # Test with-overlap cases as well.
            for k in range(1, 3):
                g = Grid(cpu_data, x_overlap=k)
                gpu_data = g.get()
                if comm.Get_rank() == 0:
                    self.assertTrue((cpu_data == gpu_data).all())

                cpu_raw = get_cpu_raw(cpu_data, k)
                self.assertTrue((cpu_raw == g._get_raw()).all())

    def test_synchronize(self):
        """ Make sure that we can make the overlap spaces accurate. """
        for case in self.cases:
            space.initialize_space(case['shape'])
            data = np.random.randn(*case['shape']).astype(case['dtype'])
            cpu_data = np.empty_like(data)
            comm.Allreduce(data, cpu_data)
            g = Grid(case['dtype'])
            self.assertRaises(TypeError, g.synchronize) # No overlap.
            # Test with-overlap cases as well.
            for k in range(1, 4):
                g = Grid(case['dtype'], x_overlap=k)

                # Overwrite entire grid
                data = np.random.randn(*case['shape']).astype(case['dtype'])
                cpu_data = np.empty_like(data)
                comm.Allreduce(data, cpu_data)
                cpu_raw_bad = get_cpu_raw(cpu_data, k)
                cpu_raw_bad[:k,:,:] += 1 # Mess up padding areas.
                cpu_raw_bad[-k:,:,:] += 1
                drv.memcpy_htod(g.data.ptr, cpu_raw_bad)

                # Prove that the data is not synchronized at this time.
                cpu_raw = get_cpu_raw(cpu_data, k)
                xx = case['shape'][0]
                gd = g._get_raw()
                self.assertTrue((gd[:k,:,:] != cpu_raw[:k,:,:]).all())
                self.assertTrue((gd[-k:,:,:] != cpu_raw[-k:,:,:]).all())

                g.synchronize() # Synchronize the overlapping data.

                # Make sure that the overlap data is accurate.
                gd = g._get_raw()
                self.assertTrue((gd[:k,:,:] == cpu_raw[:k,:,:]).all())
                self.assertTrue((gd[-k:,:,:] == cpu_raw[-k:,:,:]).all())

                comm.Barrier() # Wait for other mpi nodes to finish.


def get_cpu_raw(cpu_data, k):
    # Make sure overlapped data is accurate as well.
    xr = space.get_space_info()['x_range']
    if comm.Get_rank() == 0:
        pad_back = cpu_data[-k:,:,:]
    else:
        pad_back = cpu_data[xr[0]-k:xr[0],:,:]

    if comm.Get_rank() == comm.Get_size() - 1:
        pad_front = cpu_data[:k,:,:]
    else:
        pad_front = cpu_data[xr[1]:xr[1]+k,:,:]

    return np.concatenate((pad_back, cpu_data[xr[0]:xr[1],:,:], \
                                pad_front), axis=0)


if __name__ == '__main__':
    unittest.main()

