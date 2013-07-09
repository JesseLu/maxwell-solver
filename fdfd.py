import h5py
import numpy as np
import maxwell_ops_lumped
from solvers import bicg
from gce.grid import Grid
from mpi4py.MPI import COMM_WORLD as comm
import time, sys, tempfile, os

from pycuda import driver

def simulate(name, check_success_only=False):
    """ Read simulation from input file, simulate, and write out results. """

    # Reset the environment variables pointing to the temporary directory.
    tempfile.tempdir = '/tmp'

    # Create the reporter function.
    write_status = lambda msg: open(name + '.status', 'a').write(msg)
    if comm.Get_rank() == 0:
        # write_status('EXEC initializing\n')
        def rep(err):
            write_status('%e\n' % err)
    else: # No reporting needed for non-root nodes.
        def rep(err):
            pass
            
    # Get input parameters.
    params = get_parameters(name)

    # Define operations needed for the lumped bicg operation.
    # b, x, ops, aux_ops = maxwell_ops_lumped.ops(params)
    b, x, ops, post_cond  = maxwell_ops_lumped.ops(params)

    # Solve!
    start_time = time.time()
    rep.stime = start_time
    driver.start_profiler()
    x, err, success = bicg.solve_symm_lumped(b, x=x, \
                                            max_iters=params['max_iters'], \
                                            reporter=rep, \
                                            err_thresh=params['err_thresh'], \
                                            **ops)
    driver.stop_profiler()
#     # Last update to status file.
#     if comm.Get_rank() == 0:
#         write_status("Convergence %s in %1.1f seconds\n" % \
#             (("success" if success else "FAIL"), \
#             ((time.time() - start_time))))

    if check_success_only: # Don't write output, just see if we got a success.
        return success


    # post_cond(x) # Apply "postconditioner" to x.

#     # Calculate H-field.
#     y = ops['zeros']()
#     aux_ops['calc_H'](y, x)

    # Gather results onto root's host memory.
    result = {  'E': [E.get() for E in x], \
#                 'H': [H.get() for H in y], \
                'err': err, \
                'success': success}

    # Postcondition E.
#     # Scalar correction to the H fields.
#     if comm.Get_rank() == 0:
#         result['H'] = [(1j / params['omega']) * H for H in result['H']]
#                 
    # Write results to output file.
    if comm.Get_rank() == 0:
        result['E'] = post_cond(result['E'])
        write_results(name, result)

    return success

def get_parameters(name):
    """ Reads the simulation parameters from the input hdf5 file. """

    if comm.rank == 0:
        f = h5py.File(name + '.grid', 'r')
        files_to_delete = [name + '.grid']

        omega = np.complex128(f['omega_r'][0] + 1j * f['omega_i'][0])
        shape = tuple([int(s) for s in f['shape'][:]])

        # bound_conds = f['bound_conds'][:]

        # Function used to read in a 1D complex vector fields.
        get_1D_fields = lambda a: [(f[a+'_'+u+'r'][:] + 1j * f[a+'_'+u+'i'][:]).\
                                astype(np.complex128) for u in 'xyz']

        # Read in s and t vectors.
        s = get_1D_fields('sp')
        t = get_1D_fields('sd')

        # Read in max_iters and err_thresh.
        max_iters = int(f['max_iters'][0])
        # max_iters = 100
        err_thresh = float(f['err_thresh'][0])


        f.close() # Close file.

        # Function used to read in 3D complex vector fields.
        def get_3D_fields(a):
            field = []
            for k in range(3):
                key = name + '.' + a + '_' + 'xyz'[k]
                field.append((h5py.File(key + 'r')['data'][:] + \
                        1j * h5py.File(key + 'i')['data'][:]).astype(np.complex128))
                files_to_delete.append(key + 'r')
                files_to_delete.append(key + 'i')
            return field

#         # Read in m, e, and j fields.
#         for name in 'eJmE':
#             print comm.rank, name
#             params[name] = get_3D_fields(name)
        e = get_3D_fields('e')
        j = get_3D_fields('J')
        m = get_3D_fields('m')
        x = get_3D_fields('E')


        for filename in files_to_delete:
            os.remove(filename)

        # Do some simple pre-computation.
        for k in range(3):
            m[k] = m[k]**-1
            e[k] = omega**2 * e[k]
            j[k] = -1j * omega * j[k]

        params = {'omega': omega, 'shape': shape, \
                'max_iters': max_iters, 'err_thresh': err_thresh, \
                's': s, 't': t}
                # 'e': e, 'm': m, 'j': j, 'x': x}
    else:
        params = None

    params = comm.bcast(params)

    if comm.rank == 0:
        params['e'] = e
        params['m'] = m
        params['j'] = j
        params['x'] = x
        
    else:
        for field_name in 'emjx':
            params[field_name] = [None] * 3

    return params


def write_results(name, result):
    """ Write out the results to an hdf5 file. """

#     file = h5py.File(outfile, 'w') # Open the file.
    my_write = lambda fieldname, data: h5py.File(name + '.' + fieldname, 'w').\
                                            create_dataset('data', data=data)

    # Write out the datasets.
    for k in range(3):
        my_write('E_' + 'xyz'[k] + 'r', \
                np.real(result['E'][k]).astype(np.float32))
        my_write('E_' + 'xyz'[k] + 'i', \
                np.imag(result['E'][k]).astype(np.float32))
#             file.create_dataset(f + '_' + 'xyz'[k] + '_real', \
#                                 data=np.real(result[f][k]).astype(np.float64),
#                                 compression=1)
#             file.create_dataset(f + '_' + 'xyz'[k] + '_imag', \
#                                 data=np.imag(result[f][k]).astype(np.float64),
#                                 compression=1)
# 
#     file.create_dataset('err', data=np.float32(result['err'])) # Error log of solver.
#     file.create_dataset('success', data=result['success']) # Whether or not we succeeded.
# 
#     file.close() # Close file.


if __name__ == '__main__': # Allows calls from command line.
    simulate(sys.argv[1]) # Specify name of the job.
