import h5py
import numpy as np
import maxwell_ops_lumped
from solvers import bicg
from gce.grid import Grid
from mpi4py.MPI import COMM_WORLD as comm
import time, sys, tempfile

def simulate(name, check_success_only=False):
    """ Read simulation from input file, simulate, and write out results. """

    # Reset the environment variables pointing to the temporary directory.
    tempfile.tempdir = '/tmp'

    # Create the reporter function.
    write_status = lambda msg: open(name + '.s', 'a').write(msg)
    if comm.Get_rank() == 0:
        # write_status('EXEC initializing\n')
        def rep(err):
            write_status('Solving... %e (%1.1f seconds)\n' % (err, (time.time() - rep.stime)))
    else: # No reporting needed for non-root nodes.
        def rep(err):
            pass
            
    # Get input parameters.
    params = get_parameters(name + '.i')

    # Define operations needed for the lumped bicg operation.
    b, x, ops, aux_ops = maxwell_ops_lumped.ops(params)

    # Solve!
    start_time = time.time()
    rep.stime = start_time
    x, err, success = bicg.solve_symm_lumped(b, x=x, \
                                            max_iters=params['max_iters'], \
                                            reporter=rep, \
                                            err_thresh=params['err_thresh'], \
                                            **ops)
    # Last update to status file.
    if comm.Get_rank() == 0:
        write_status("Convergence %s in %1.1f seconds\n" % \
            (("success" if success else "FAIL"), \
            ((time.time() - start_time))))

    if check_success_only: # Don't write output, just see if we got a success.
        return success


    aux_ops['post_cond'](x) # Apply "postconditioner" to x.

    # Calculate H-field.
    y = ops['zeros']()
    aux_ops['calc_H'](y, x)

    # Gather results onto root's host memory.
    result = {  'E': [E.get() for E in x], \
                'H': [H.get() for H in y], \
                'err': err, \
                'success': success}

    # Scalar correction to the H fields.
    if comm.Get_rank() == 0:
        result['H'] = [(1j / params['omega']) * H for H in result['H']]
                
    # Write results to output file.
    if comm.Get_rank() == 0:
        write_results(name, result)

    return success

def get_parameters(infile):
    """ Reads the simulation parameters from the input hdf5 file. """

    f = h5py.File(infile, 'r')

    omega = np.complex128(f['omega_real'][0] + 1j * f['omega_imag'][0])
    # bound_conds = f['bound_conds'][:]

    # Function used to read in the 1D and 3D complex vector fields.
    get_field = lambda a: [(f[a+'_'+u+'_real'][:] + 1j * f[a+'_'+u+'_imag'][:]).\
                            astype(np.complex128) for u in 'xyz']

    # Read in s and t vectors.
    s = get_field('d_prim')
    t = get_field('d_dual')

    # Read in m, e, and j fields.
    m = get_field('mu')
    e = get_field('epsilon')
    j = get_field('J')
    x = get_field('E')

#     # Add a relatively small measure of randomness to x.
#     # Make the magnitude of the random numbers to be about 1/1000th of the
#     # maximum value of x.
#     x_mag = np.max([np.max(np.abs(xcomp)) for xcomp in x]) 
#     j_mag = np.max([np.max(np.abs(jcomp)) for jcomp in j]) 
#     if x_mag == 0: 
#         rand_mag = 1e-3 * j_mag
#     else:
#         rand_mag = 1e-3 * x_mag
# 
#     # Add the randomness.
#     x = [xcomp + rand_mag * (np.random.randn(*xcomp.shape) + \
#                             1j*np.random.randn(*xcomp.shape)) for xcomp in x]

    # Read in max_iters and err_thresh.
    max_iters = int(f['max_iters'][0])
    err_thresh = float(f['err_thresh'][0])

    f.close() # Close file.

    # Do some simple pre-computation.
    for k in range(3):
        m[k] = m[k]**-1
        e[k] = omega**2 * e[k]
        j[k] = -1j * omega * j[k]

    # Return all inputs as a dictionary.
    return {'omega': omega, 's': s, 't': t, \
            'm': m, 'e': e, 'j': j, 'x': x, \
            'max_iters': max_iters, 'err_thresh': err_thresh}

def write_results(name, result):
    """ Write out the results to an hdf5 file. """

#     file = h5py.File(outfile, 'w') # Open the file.
    my_write = lambda fieldname, data: h5py.File(name + '.' + fieldname, 'w').\
                                            create_dataset('data', data=data)

    # Write out the datasets.
    for f in ['E', 'H']:
        for k in range(3):
            my_write(f + 'xyz'[k] + '_real', \
                    np.real(result[f][k]).astype(np.float32))
            my_write(f + 'xyz'[k] + '_imag', \
                    np.imag(result[f][k]).astype(np.float32))
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
