import cProfile
import functools
import os
import pstats
import time

from src.utils import log


def profiler(func=None, timer="all"):
    """
    Profiles a function to get the time each elements takes
    """

    def _decorate(_func):
        @functools.wraps(_func)
        def decorator_profiler(*args, **kwargs):
            log('event', f"Running function {_func.__name__}() with profiling")
            # print(f"Running function {func.__name__}() with profiling")
            et = ct = 0
            pr = cProfile.Profile()
            pr.enable()
            if timer.lower() == "cpu":
                et = time.time()
                ct = os.times()[0]
            val = _func(*args, **kwargs)
            pr.disable()
            if timer.lower() == "cpu":
                log('info', f"EXECUTION TIME : {time.time() - et:.4f} seconds.")
                log('info', f"CPU TIME       : {os.times()[0] - ct:.4f} seconds.")
            pstats.Stats(pr).strip_dirs().sort_stats("time").print_stats(10)
            return val

        return decorator_profiler

    if func:
        return _decorate(func)

    return _decorate


@profiler
def heavy(load=25, timeout=10):
    os.system(f"stress -c {load}  -t {timeout}")


if __name__ == '__main__':
    heavy()
