'''
Author: huzx huzx@zetyun.com
Date: 2023-07-24 11:11:27
LastEditors: huzx huzx@zetyun.com
LastEditTime: 2023-07-24 14:00:37
FilePath: /kb_server/utils/timing.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import time
from collections.abc import Callable
from collections.abc import Generator
from typing import Any
from typing import cast
from typing import TypeVar
from typing import Optional

from utils.logger import setup_logger

logger = setup_logger()

F = TypeVar("F", bound=Callable)

def log_function_time(
    func_name: Optional[str] = None,
) -> Any:
    """Build a timing wrapper for a function. Logs how long the function took to run.
    Use like:

    @log_function_time()
    def my_func():
        ...
    """
    def timing_wrapper(func: F) -> F:
        def wrapped_func(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            time_cost = round(time.time() - start_time, 3)
            logger.info(
                f"Metric statics {func_name or func.__name__} took {time_cost} seconds"
            )
            return result

        return cast(F, wrapped_func)

    return timing_wrapper


def log_generator_function_time(
    func_name: Optional[str] = None,
) -> Any:
    """Build a timing wrapper for a function which returns a generator.
    Logs how long the function took to run.
    Use like:

    @log_generator_function_time()
    def my_func():
        ...
        yield X
        ...
    """

    def timing_wrapper(func: F) -> F:
        def wrapped_func(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            yield from func(*args, **kwargs)
            time_cost = round(time.time() - start_time, 3)
            logger.info(
                f"{func_name or func.__name__} took {time_cost} seconds"
            )

        return cast(F, wrapped_func)

    return timing_wrapper


