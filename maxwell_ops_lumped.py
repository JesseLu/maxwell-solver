""" Implements the operations needed to solve Maxwell's equations in 3D. """

import numpy as np
from jinja2 import Environment, PackageLoader, Template
from gce.space import initialize_space, get_space_info
from gce.grid import Grid
from gce.const import Const
from gce.out import Out
from gce.kernel import Kernel

# Execute when module is loaded.
# Load the jinja environment.
jinja_env = Environment(loader=PackageLoader(__name__, 'kernels'))

def ops(params):
    """ Define the operations that specify the symmetrized, lumped problem. """

    # Initialize the space.
    initialize_space(params['shape'])

    dtype = np.complex128

    b = [Grid(dtype(f), x_overlap=1) for f in params['j']]
    x = [Grid(dtype(f), x_overlap=1) for f in params['x']]

    pre_cond, post_cond = conditioners(params, dtype)
    pre_cond(b) # "Precondition" b.

    # Return b, the lumped operations needed for the bicg algorithm, and
    # the postconditioner to obtain the "true" x.
    return  b, x, \
            {'zeros': lambda: [Grid(dtype, x_overlap=1) for k in range(3)], \
            'rho_step': rho_step(dtype), \
            'alpha_step': alpha_step(params, dtype)}, \
            post_cond
#             {'post_cond': post_cond, \
#             'calc_H': calc_H(shape, params, dtype)}
            

def rho_step(dtype):
    """ Return the function to execute the rho step of the bicg algorithm. """

    # Code for the rho step function.
    code = Template("""
        if (_in_global) {
            x0(0,0,0) = x0(0,0,0) + alpha * p0(0,0,0);
            x1(0,0,0) = x1(0,0,0) + alpha * p1(0,0,0);
            x2(0,0,0) = x2(0,0,0) + alpha * p2(0,0,0);
            {{ type }} s0 = r0(0,0,0) - alpha * v0(0,0,0);
            {{ type }} s1 = r1(0,0,0) - alpha * v1(0,0,0);
            {{ type }} s2 = r2(0,0,0) - alpha * v2(0,0,0);
            rho += (s0 * s0) + (s1 * s1) + (s2 * s2);
            err +=  (real(s0) * real(s0)) + \
                    (imag(s0) * imag(s0)) + \
                    (real(s1) * real(s1)) + \
                    (imag(s1) * imag(s1)) + \
                    (real(s2) * real(s2)) + \
                    (imag(s2) * imag(s2));
            r0(0,0,0) = s0;
            r1(0,0,0) = s1;
            r2(0,0,0) = s2;
        } """).render(type=_get_cuda_type(dtype))
                    
    # Compile the code.
    grid_names = [A + i for A in ['p', 'r', 'v', 'x'] for i in ['0', '1', '2']]
    rho_fun = Kernel(code, \
                    ('alpha', 'number', dtype), \
                    ('rho', 'out', dtype), \
                    ('err', 'out', dtype), \
                    *[(name, 'grid', dtype) for name in grid_names], \
                    shape_filter='skinny')

    # Temporary values that are needed.
    rho_out = Out(dtype)
    err_out = Out(dtype)

    # Define the actual function.
    def rho_step(alpha, p, r, v, x):
        rho_fun(dtype(alpha), rho_out, err_out, *(p + r + v + x), \
                post_sync=r) # r must be post-synced for upcoming alpha step.
        return rho_out.get(), np.sqrt(err_out.get())

    return rho_step


def alpha_step(params, dtype): 
    """ Define the alpha step function needed for the bicg algorithm. """
    num_shared_banks = 6 

    # Render the pre-loop and in-loop code.
    cuda_type = _get_cuda_type(dtype)
    code_allpre = jinja_env.get_template('alpha_allpre.cu').\
                    render(dims=params['shape'], \
                            type=cuda_type, \
                            mu_equals_1=False, \
                            full_operator=True)

    # Grid input parameters.
    grid_params = [(A + i, 'grid', dtype) for A in ['P', 'P1', 'R', 'V', 'e', 'm'] \
                                            for i in ['x', 'y', 'z']]

    # Const input parameters.
    const_names = ('sx0', 'sy0', 'sz0', 'sx1', 'sy1', 'sz1') + \
                    ('sqrt_sx0', 'sqrt_sy0', 'sqrt_sz0', \
                    'sqrt_sx1', 'sqrt_sy1', 'sqrt_sz1')
    const_sizes = params['shape'] * 4
    const_params = [(const_names[k], 'const', dtype, const_sizes[k]) \
                        for k in range(len(const_sizes))]

    # Compile.
    alpha_fun = Kernel('', \
                    ('beta', 'number', dtype), \
                    ('alpha_denom', 'out', dtype), \
                    *(grid_params + const_params), \
                    pre_loop=code_allpre, \
                    padding=(1,1,1,1), \
                    smem_per_thread=num_shared_banks*16, \
                    shape_filter='square')

    # Temporary variables.
    alpha_denom_out = Out(dtype)
    p_temp = [Grid(dtype, x_overlap=1) for k in range(3)] # Used to swap p.

    # Grid variables.
    e = [Grid(dtype(f), x_overlap=1) for f in params['e']]
    m = [Grid(dtype(f), x_overlap=1) for f in params['m']] # Optional.

    # Constant variables.
    sc_pml_0 = [Const(dtype(s**-1)) for s in params['s']]
    sc_pml_1 = [Const(dtype(t**-1)) for t in params['t']]
    sqrt_sc_pml_0 = [Const(dtype(np.sqrt(s**-1))) for s in params['s']]
    sqrt_sc_pml_1 = [Const(dtype(np.sqrt(t**-1))) for t in params['t']]

    # Define the function
    def alpha_step(rho_k, rho_k_1, p, r, v):
        # Execute cuda code.
        # Notice that p_temp and v are post_synced.
        alpha_fun(dtype(rho_k/rho_k_1), alpha_denom_out, \
                    *(p + p_temp + r + v + e + m + \
                        sc_pml_0 + sc_pml_1 + sqrt_sc_pml_0 + sqrt_sc_pml_1), \
                    post_sync=p_temp+v)
        p[:], p_temp[:] = p_temp[:], p[:] # Deep swap.

        return rho_k / alpha_denom_out.get() # The value of alpha.

    return alpha_step


def conditioners(params, dtype): 
    """ Form the functions for both the preconditioner and postconditioner. """

    # Code for the post step function.
    code = """
        if (_in_global) {
            Ex(0,0,0) *= tx1(_X) * ty0(_Y) * tz0(_Z);
            Ey(0,0,0) *= tx0(_X) * ty1(_Y) * tz0(_Z);
            Ez(0,0,0) *= tx0(_X) * ty0(_Y) * tz1(_Z);
        } """
                    
    # Form the Gird parameters.
    grid_names = ['Ex', 'Ey', 'Ez']
    grid_params = [(name, 'grid', dtype) for name in grid_names]

    # Form the Const parameters
    const_names = ('tx0', 'ty0', 'tz0', \
                    'tx1', 'ty1', 'tz1')
    const_sizes = params['shape'] * 2
    const_params = [(const_names[k], 'const', dtype, const_sizes[k]) \
                        for k in range(len(const_sizes))]

    # Compile the code.
    post_fun = Kernel(code, *(grid_params + const_params), \
                    shape_filter='skinny')

    # Consts that are used.
    sqrt_sc_pml_0 = [Const(dtype(np.sqrt(s)**1)) for s in params['s']]
    sqrt_sc_pml_1 = [Const(dtype(np.sqrt(t)**1)) for t in params['t']]
    inv_sqrt_sc_pml_0 = [Const(dtype(np.sqrt(s)**-1)) for s in params['s']]
    inv_sqrt_sc_pml_1 = [Const(dtype(np.sqrt(t)**-1)) for t in params['t']]

    # Define the actual functions.

    def pre_step(x):
        post_fun(*(x + sqrt_sc_pml_0 + sqrt_sc_pml_1))

    def post_step(x):
        post_fun(*(x + inv_sqrt_sc_pml_0 + inv_sqrt_sc_pml_1))

    return pre_step, post_step


def _get_cuda_type(dtype):
    """ Convert numpy type into cuda type. """
    if dtype is np.complex64:
        return 'pycuda::complex<float>'
    elif dtype is np.complex128:
        return 'pycuda::complex<double>'
    else:
        raise TypeError('Invalid dtype.')
