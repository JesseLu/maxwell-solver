import unittest
import space
from mpi4py.MPI import COMM_WORLD as comm

class TestSpace(unittest.TestCase):
    def setUp(self):
        """ Uninitialize the module variable for the global space. """
        # space.destroy_space()

    def test_ecc_disabled(self):
        """ Make sure ECC is disabled. """
        space.initialize_space((100, 2, 3))
        self.assertTrue(space.get_space_info()['ecc_enabled'] == False, \
            'ECC enabled! Should be disabled for best performance.')

    def test_three_elements(self):
        """ Make sure only 3D spaces can be successfully created. """
        self.assertRaises(TypeError, space.initialize_space, (100,))
        self.assertRaises(TypeError, space.initialize_space, (100, 2))
        self.assertRaises(TypeError, space.initialize_space, (100, 2, 3, 4))
        self.assertRaises(TypeError, space.initialize_space, (100, 2, 3, 4, 5))
        space.initialize_space((100, 2, 3))

    def test_integer_elements(self):
        """ Make sure only spaces with integer size dimensions are ok. """
        self.assertRaises(TypeError, space.initialize_space, (100, 2.1, 3))

    def test_positive_elements(self):
        """ Make sure only positive elements are allowed. """
        self.assertRaises(TypeError, space.initialize_space, (100, -2, 3))

#     def test_stencil(self):
#         """ Make sure stencil only allows for single non-negative integers. """
#         self.assertRaises(TypeError, space.initialize_space, (100, 2, 3), \
#                             stencil=-1)
#         self.assertRaises(TypeError, space.initialize_space, (100, 2, 3), \
#                             stencil=(1, 2))

    def test_get_info(self):
        """ Test the get_space_info function. """
#         # We should get an error if we haven't initialized a space yet.
#         self.assertRaises(TypeError, space.get_space_info)

        shape = (100,2,3)
        space.initialize_space(shape)
        info = space.get_space_info()
        self.assertEqual(info['shape'], shape)
        # self.assertEqual(info['stencil'], stencil)

    def test_partition(self):
        """ Make sure the x_ranges span the entire space without any gaps. """
        shapes = ((200,30,10), (33,10,10), (130,5,5), (111,2,2))
        for shape in shapes:
            space.initialize_space(shape)
            x = comm.gather(space.get_space_info()['x_range'])
            if comm.Get_rank() == 0:
                self.assertEqual(x[0][0], 0)
                self.assertEqual(x[-1][-1], space.get_space_info()['shape'][0])
                for k in range(len(x)-1):
                    self.assertEqual(x[k][1], x[k+1][0])

if __name__ == '__main__':
    unittest.main()
