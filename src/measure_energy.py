import os
import time

from src.conf import IDLE_POWER
from src.utils import log


def measure_energy(func):
    """
    Measure the energy consumption during the execution of the decorated function
    """

    def decorator_measure_energy(*args, **kwargs):
        pre_execution()
        t = time.time()
        log('event', f"Starting energy estimation of: {func.__name__}()")
        val: str = func(*args, **kwargs)
        t = time.time() - t
        log('info', f"{func.__name__}() took {t:.2f} seconds to finish.")
        post_execution(t)

        return val

    return decorator_measure_energy


def pre_execution():
    os.system(f"src/scripts/pre_execution.sh {os.getpid()} > /dev/null")


def post_execution(t, filename="/tmp/measure_energy.tmp"):
    if t < 1:
        log('warning', f"Energy estimation works only with execution time greater than 1 second")
        return
    os.system("src/scripts/post_execution.sh > /dev/null")
    tmp_file = "/tmp/measure_energy.tmp"
    powerstat_file = None
    avg = None
    std = None
    # get powerstat file
    with open(tmp_file) as f:
        for line in f:
            if 'File:' in line:
                powerstat_file = line.rstrip().split(' ')[-1]
    os.remove(tmp_file)
    # read powerstat file
    with open(powerstat_file) as f:
        for line in f:
            if "Average" in line:
                avg = line.strip().split('  ')[-2:]
            elif "StdDev" in line:
                std = line.strip().split('  ')[-2:]
    os.remove(powerstat_file)
    # print the results // idle_state = 13.13
    program_power = float(avg[0]) - IDLE_POWER
    log('result', f"Execution time          : {t:.2f} Seconds")
    log('', f"Average Power usage           : {avg[0]} Watts; (StdDev of {std[0]} Watts)")
    log('', f"Average Program power usage   : {program_power:.2f} Watts; (StdDev of {std[0]} Watts)")
    log('', f"Average Frequency             : {avg[1].strip()}; (StdDev of {std[1]})")
    log('', f"Average Energy                : {t * float(avg[0]):.2f} J; (StdDev of {t * float(std[0]):.2f} J)")
    log('', f"Average Program Energy        : {t * program_power:.2f} J; (StdDev of {t * float(std[0]):.2f} J)")
