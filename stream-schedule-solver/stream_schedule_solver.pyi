from typing import Callable, TypeVar, Any
from mdsdf import Sdf2D

E = TypeVar('E') 

class Optimizer[E]:
    def __init__(
        memory_sizes: list[int],
        name_fn: Callable[[E], str],
        execution_time_fn: Callable[[E], int],
        tensor_size_fn: Callable[[E], int],
        memory_id_fn: Callable[[E], int],
        processor: Callable[[E], set[Any]],
        sdf: Sdf2D[E],
    ):
        pass