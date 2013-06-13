import numpy as np 
__shapes = [(41,23,31), (48,59,67), (100,100,100)]
__dtypes = [np.float32, np.float64, np.complex64, np.complex128] 
cases = [{'shape':s, 'dtype':t} for s in __shapes for t in __dtypes]
