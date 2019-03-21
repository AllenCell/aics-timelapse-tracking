import time


def report_run_time(fn):
    """Decorator to report function execution time."""
    def wrapper(*args, **kwargs):
        tic = time.time()
        retval = fn(*args, **kwargs)
        toc = time.time()
        name = fn.__module__ + '.' + fn.__qualname__
        print(f'{name} executed in {toc - tic:.2f} s.')
        return retval
    return wrapper
