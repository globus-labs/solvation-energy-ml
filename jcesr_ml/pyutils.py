"""Utility functions for general Python tasks

These functions, generally, are things I am surprised I could not find elsewhere"""


def sequential(funcs, x):
    """Execute a list of functions where the output of one function is the input of the next

    Args:
        funcs ([functions]): List of functions. Functions are executed from front of the
            list to the back (i.e., funcs[0] is the first function.
        x: Input to the first function
    Returns:
        Output of the last function
    """

    for f in funcs:
        x = f(x)
    return x
