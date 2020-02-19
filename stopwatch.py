
import time


class stopwatch:
    """A class for measuring execution time."""

    def __init__(self, name="Execution", verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        print(f'{self.name} started...')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        delta = self.end - self.start
        if self.verbose:
            print(f'{self.name} completed in {round(delta, 2)} s.')


def stopwatch_dec(func):
    """An alternative decorator for measuring the elapsed time."""

    def wrapper(*args, **kwargs):
        start = time.time()
        print(f'\n{self.name} started.')
        results = func(*args, **kwargs)
        delta = time.time() - start
        print(f'\n{self.name} completed in {round(delta, 1)} s.')
        return results
    return wrapper
