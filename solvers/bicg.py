import numpy as np

def _axby(a, x, b, y):
    """ Default axby routine. """
    y[:] = a * x + b * y


def _nochange(x, y):
    """ Default multA and multAT routine, simply sets y = x. """
    _axby(1, x, 0, y)


def solve_asymm(b, x=None, x_hat=None, multA=_nochange, multAT=_nochange, 
                norm=np.linalg.norm, dot=np.dot, axby=_axby, \
                zeros=None, \
                eps=1e-6, max_iters=1000):
    """ Bi-conjugate gradient solve of square, non-symmetric system.

    Input variables:
    b -- the problem to be solved is A * x = b.

    Keyword variables:
    x -- initial guess of x, default value is 0.
    x_hat -- initial guess for x_hat, default value is 0.
    multA -- multA(x) calculates A * x and returns the result,
             default: returns x.
    multAT -- multAT(x) calculates A^T * x and returns the result.
              default: returns x.
    dot -- dot(x, y) calculates xT * y and returns the result, 
           default: numpy.dot().
    axby -- axby(a, x, b, y) calculates a * x + b * y and 
            stores the result in y, default y[:] = a * x + b * y.
    copy -- copy(x) returns a copy of x, default numpy.copy().
    eps -- the termination error is determined by eps * ||b|| (2-norm),
           default 1e-6.
    max_iters -- maximum number of iterations allowed, default 1000.

    Output variables:
    x -- the apprximate answer of A * x = b.
    err -- a numpy array with the error value at every iteration.
    success -- True if convergence was successful, False otherwise.
    """

    # TODO
    # ----
    # *   Check for breakdown condition.
    # *   Notify user if we weren't able to beat term_err.

    if zeros is None: # Default version of the zeros operation.
        def zeros():
            return np.zeros_like(b)

    if x is None: # Default value of x is 0.
        x = zeros()
        axby(0, b, 0, x)

    if x_hat is None: # Default value for x_hat is 0.
        x_hat = zeros()
        axby(0, b, 0, x_hat)

    # r = b - A * x.
    r = zeros()
    multA(x, r)
    axby(1, b, -1, r)

    # r_hat = b - A * x_hat
    r_hat = zeros()
    multAT(x, r_hat)
    axby(1, b, -1, r_hat)

    # p = r, p_hat = r_hat.
    p = zeros()
    axby(1, r, 0, p)
    p_hat = zeros()
    axby(1, r_hat, 0, p_hat)

    # Initialize v, v_hat. Used to store A * p, AT * p_hat.
    v = zeros() # Don't need the values, this is an "empty" copy.
    v_hat = zeros()

    rho = np.zeros(max_iters).astype(np.complex128) # Related to error.
    err = np.zeros(max_iters).astype(np.float64) # Error.
    term_err = eps * norm(b) # Termination error value.

    for k in range(max_iters):

        # Compute error and check termination condition.
        err[k] = norm(r)
        print k, err[k]

        if err[k] < term_err: # We successfully converged!
            return x, err[:k+1], True
	elif np.isnan(err[k]): # Hopefully this will never happen.
            return x, err[:k+1], False 

        
        # rho = r_hatT * r.
        rho[k] = dot(r_hat, r)

#         if abs(rho[k]) < 1e-15: # Breakdown condition.
#             raise ArithmeticError('Breakdown')

        multA(p, v)
        multAT(p_hat, v_hat)

        # alpha = rho / (p_hatT * v).
        alpha = rho[k] / dot(p_hat, v)

        # x += alpha * p, x_hat += alpha * p_hat.
        axby(alpha, p, 1, x)
        axby(alpha, p_hat, 1, x_hat)

        # r -= alpha * v, r -= alpha * v_hat. 
        axby(-alpha, v, 1, r)
        axby(-alpha, v_hat, 1, r_hat)

        # beta = (r_hatT * r) / rho.
        beta = dot(r_hat, r) / rho[k]

        # p = r + beta * p, p_hat = r_hat + beta * p_hat.
        axby(1, r, beta, p)
        axby(1, r_hat, beta, p_hat)

    # Return the answer, and the progress we made.
    return x, err, False
        

def solve_symm(b, x=None, multA=_nochange, 
                norm=np.linalg.norm, dot=np.dot, \
                axby=_axby, zeros=None, eps=1e-6, max_iters=1000):
    """ Bi-conjugate gradient solve of square, symmetric system.

    Input variables:
    b -- the problem to be solved is A * x = b.

    Keyword variables:
    x -- initial guess of x, default value is 0.
    multA -- multA(x) calculates A * x and returns the result,
             default: returns x.
    dot -- dot(x, y) calculates xT * y and returns the result, 
           default: numpy.dot().
    axby -- axby(a, x, b, y) calculates a * x + b * y and 
            stores the result in y, default y[:] = a * x + b * y.
    zeros -- zeros() creates a zero-initialized vector. 
    eps -- the termination error is determined by eps * ||b|| (2-norm),
           default 1e-6.
    max_iters -- maximum number of iterations allowed, default 1000.

    Output variables:
    x -- the approximate answer of A * x = b.
    err -- a numpy array with the error value at every iteration.
    success -- True if convergence was successful, False otherwise.
    """

    # TODO
    # ----
    # *   Check for breakdown condition.
    # *   Notify user if we weren't able to beat term_err.

    if zeros is None: # Default version of the zeros operation.
        def zeros():
            return np.zeros_like(b)

    if x is None: # Default value of x is 0.
        x = zeros()

    # r = b - A * x.
    r = zeros()
    multA(x, r)
    axby(1, b, -1, r)

    # p = r.
    p = zeros()
    axby(1, r, 0, p)

    # Initialize v. Used to store A * p.
    v = zeros() # Don't need the values, this is an "empty" copy.
    v_hat = zeros()

    rho = np.zeros(max_iters).astype(np.complex128) # Related to error.
    err = np.zeros(max_iters).astype(np.float64) # Error.
    term_err = eps * norm(b) # Termination error value.

    for k in range(max_iters):

        # Compute error and check termination condition.
        err[k] = norm(r)
        print k, err[k]

        if err[k] < term_err: # We successfully converged!
            return x, err[:k+1], True
        
        # rho = r^T * r.
        rho[k] = dot(r, r)

#         if abs(rho[k]) < 1e-15: # Breakdown condition.
#             raise ArithmeticError('Breakdown')

        multA(p, v)

        # alpha = rho / (p^T * v).
        alpha = rho[k] / dot(p, v)

        # x += alpha * p.
        axby(alpha, p, 1, x)

        # r -= alpha * v.
        axby(-alpha, v, 1, r)

        # beta = (r^T * r) / rho.
        beta = dot(r, r) / rho[k]

        # p = r + beta * p.
        axby(1, r, beta, p)

    # Return the answer, and the progress we made.
    return x, err, False
        

def solve_symm_lumped(r, x=None, rho_step=None, alpha_step=None, zeros=None, \
                    err_thresh=1e-6, max_iters=1000, reporter=lambda err: None):
# Note: r is used instead of b in the input parameters of the function.
# This is in order to initialize r = b, and to inherently disallow access to 
# b from within this function.
    """ Lumped bi-conjugate gradient solve of a symmetric system.

    Input variables:
    b -- the problem to be solved is A * x = b.

    Keyword variables:
    x -- initial guess of x, default value is 0.
    rho_step -- rho_step(alpha, p, r, v, x) updates r and x, and returns
        rho and the error. Specifically, rho_step performs:
            x = x + alpha * p
            r = r - alpha * v
            rho_(k+1) = (r dot r)
            err = (conj(r) dot r)
    alpha_step -- alpha_step(rho_k, rho_(k-1), p, r, v) updates p and v, and 
        returns alpha. Specifically, alpha_step performs:
            p = r + (rho_k / rho_(k-1)) * p
            v = A * p
            alpha = rho_k / (p dot v)
    zeros -- zeros() creates a zero-initialized vector. 
    err_thresh -- the relative error threshold, default 1e-6.
    max_iters -- maximum number of iterations allowed, default 1000.
    reporter -- function to report progress.

    Output variables:
    x -- the approximate answer of A * x = b.
    err -- a numpy array with the error value at every iteration.
    success -- True if convergence was successful, False otherwise.
    """

    # TODO
    # ----
    # *   Check for breakdown condition.

    # Initialize variables. 
    # Note that r = b was "initialized" in the function declaration.

    # Initialize x = 0, if defined.
    if x is None: # Default value of x is 0.
        x = zeros()

    # Initialize v = Ax.
    v = zeros()
    alpha_step(1, 1, x, zeros(), v) # Used to calculate v = Ax.

    p = zeros() # Initialize p = 0.
    alpha = 1 # Initial value for alpha.

    rho = np.zeros(max_iters).astype(np.complex128)
    rho[-1] = 1 # Silly trick so that rho[k-1] for k = 0 is defined.

    err = np.zeros(max_iters).astype(np.float64) # Error.

    temp, b_norm = rho_step(0, p, r, v, x) # Calculate norm of b, ||b||.

    print 'b_norm check: ', b_norm # Make sure this isn't something bad like 0.

    for k in range(max_iters):

        print 'rho  ', k,
        rho[k], err0 = rho_step(alpha, p, r, v, x) 
        err[k] = err0 / b_norm # Relative error.

        # Check termination condition.
        reporter(err[k])
        if err[k] < err_thresh: # We successfully converged!
            return x, err[:k+1], True

        print 'alpha', k,  
        alpha = alpha_step(rho[k], rho[k-1], p, r, v)

    # Return the answer, and the progress we made.
    return x, err, False
        
