""" Demonstrates the basic mpi4py techniques that gce uses.

To run use something like:
    mpirun -n 2 -host raven1,raven1 python simple_mpi_test.py
"""

from mpi4py import MPI
import pycuda.driver as drv
import pycuda.gpuarray as ga
import sys
import numpy as np
import unittest

class SimpleMpiTest(unittest.TestCase):
    def test1(self):
        # Test initialization.
        ctx = choose_gpu()

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Test scatter (for array distribution).
        big_data = [np.float64(np.arange(1, 101) + k * 100) for k in range(size)]
        t = comm.scatter(big_data)
        t_gpu = ga.to_gpu(t) 
        # print rank, ga.sum(t_gpu)

        # Test point-to-point communication (for array synchronization).
        forw, curr, back = (rank+1)%size, rank, (rank-1)%size
        s = np.empty_like(t)
        comm.Send(t, dest=forw, tag=forw+34)
        comm.Recv(s, source=back, tag=curr+34)
        # print rank, np.sum(s)

        # Non-blocking point-to-point.
        s = np.empty_like(t)
        comm.Isend(t, dest=forw, tag=forw+35)
        req = comm.Irecv(s, source=back, tag=curr+35)
        req.Wait()
        # print rank, np.sum(s)

        # Test reduce communication (for sum computation).
        x = np.empty(1, dtype=np.float64)
        comm.Allreduce(np.sum(t), x)
        # print rank, x

        y = np.empty(2, dtype=np.float64)
        comm.Allreduce(np.array([np.sum(t), np.min(s)]), y)
        # print rank, y

        # Test gather communication (for array unification).
        r = comm.gather(t) 
        # if rank == 0:
            # print sum([np.sum(f) for f in r])


        ctx.pop()


def choose_gpu():
    # Find out how many GPUs are available to us on this node.
    drv.init()
    num_gpus = drv.Device.count()

    # Figure out the names of the other hosts.
    rank = MPI.COMM_WORLD.Get_rank() # Find out which process I am.
    name = MPI.Get_processor_name() # The name of my node.
    hosts = MPI.COMM_WORLD.allgather(name) # Get the names of all the other hosts

    # Figure out our precendence on this node.

    # Make sure the number of hosts and processes are equal.
    num_processes = MPI.COMM_WORLD.Get_size()
    if (len(hosts) is not num_processes):
        raise TypeError('Number of hosts and number of processes do not match.')


    # Make sure the name of my node matches.
    if (name != hosts[rank]):
        # print name, hosts[rank]
        raise TypeError('Hostname does not match.')

    # Find out which GPU to take.
    gpu_id = hosts[0:rank].count(name)
    if gpu_id >= num_gpus:
        raise TypeError('No GPU available.')

#     sys.stdout.write("On %s: %d/%d taking gpu %d/%d.\n" % \
#                         (name, rank, num_processes, gpu_id, num_gpus))
    
    # Make and return a context on the device.
    return drv.Device(gpu_id).make_context() 


if __name__ == '__main__':
    unittest.main()
