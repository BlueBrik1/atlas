"""Numba accelerates the same scoring loops used by the portable demo."""
try:
    from numba import njit
except ModuleNotFoundError as error:
    if error.name != "numba":
        raise

    def njit(*, cache):
        def keep_python_function(function):
            return function
        return keep_python_function
