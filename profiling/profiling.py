import functools

from pyinstrument.profiler import Profiler


def profile_function(output_file="profile.html"):
    """
    Profiles a function execution time.

    Parameters
    ----------
    output_file: file to write profile output. Defaults to "profile.html".
    """

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            profiler = Profiler()
            profiler.start()
            result = function(*args, **kwargs)
            profiler.stop()
            output = profiler.output_html()
            with open(output_file, "w") as f:
                f.write(output)
            return result

        return wrapper

    return decorator
