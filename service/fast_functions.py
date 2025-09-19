import numpy as np
import numba as nb


@nb.njit
def numba_histogram(array, bins, hist_range=None):
    return np.histogram(array, bins, hist_range)


@nb.njit(nb.float64[:, :](nb.float64[:, :], nb.float64[:, :]))
def numba_dot(A, B):
    m, n = A.shape
    p = B.shape[1]

    C = np.zeros((m, p))

    for i in range(0, m):
        for j in range(0, p):
            for k in range(0, n):
                C[i, j] += A[i, k] * B[k, j]
    return C


# adapted from https://stackoverflow.com/questions/41648058/what-is-the-difference-between-import-numpy-and-import-math
@nb.njit
def normpdf(x, mean=0, std=1):
    var = np.power(std, 2)
    return np.exp(-np.power(x - mean, 2) / (2 * var)) / np.sqrt(2 * np.pi * var)


@nb.njit
def normcdf(x, mean=0, std=1, min_x=-10, dx=0.001):
    var = np.power(std, 2)
    array_input = np.arange(min_x, x + dx, dx)
    array_solutions = np.zeros(mean.shape[0])
    for i in range(mean.shape[0]):
        array_solutions[i] = np.sum(np.exp(-np.power(array_input - mean[i], 2) / (2 * var[i])) / np.sqrt(2 * np.pi * var[i])) * dx
    return array_solutions

@nb.njit
def normcdf_single_value(x, mean=0, std=1, min_x=-10, dx=0.001):
    var = np.power(std, 2)
    array_input = np.arange(min_x, x + dx, dx)
    return np.sum(np.exp(-np.power(array_input - mean, 2) / (2 * var)) / np.sqrt(2 * np.pi * var)) * dx


@nb.njit
def solve_fe(F, f, input_signal):
    for i_time in range(len(input_signal)-1):
        f[i_time + 1, :] = F[i_time, :, :] @ f[i_time, :]
    return f

@nb.njit
def solve_fe_constant(F, f, duration_input_signal, dt, inactive_time):
    for i_time in range(int(duration_input_signal/dt)-1):
        if i_time * dt < inactive_time:
            f[i_time + 1, :] = f[i_time, :]
        else:
            f[i_time + 1, :] = F[:, :] @ f[i_time, :]
    return f

def solve_be(F, f, input_signal, xi):
    F[F<0] = 0
    F_actual = 2 * np.stack([np.eye(len(xi))]*len(input_signal)) - F
    return _solve_be(F_actual, f, input_signal)

@nb.njit
def _solve_be(F_actual, f, input_signal):
    for i_time in range(len(input_signal)-1):
        f[i_time + 1, :] = np.linalg.solve(F_actual[i_time, :, :], f[i_time, :])
    return f

def solve_be_constant(F, f, duration_input_signal, dt, xi, inactive_time):
    F[F<0] = 0
    F_actual = 2 * np.eye(len(xi)) - F
    return _solve_be_constant(F_actual, f, duration_input_signal, dt, inactive_time)

@nb.njit
def _solve_be_constant(F_actual, f, duration_input_signal, dt, inactive_time):
    for i_time in range(int(duration_input_signal/dt)-1):
        if i_time * dt < inactive_time:
            f[i_time + 1, :] = f[i_time, :]
        else:
            f[i_time + 1, :] = np.linalg.solve(F_actual[:, :], f[i_time, :])
    return f


# adapted from https://stackoverflow.com/questions/47243190/numpy-arange-how-to-make-precise-array-of-floats
@nb.njit
def safe_arange(start, stop, step):
    return step * np.arange(start / step, stop / step)


# count elements in a dictionary with arbitrary number of nested levels
def count_entries_in_dict(my_dict, c=0):
    for my_key in my_dict:
        if isinstance(my_dict[my_key], dict):
            c = count_entries_in_dict(my_dict[my_key], c)
        elif isinstance(my_dict[my_key], np.ndarray) or isinstance(my_dict[my_key], list):
            c += len(my_dict[my_key])
        else:
            c += 1
    return c