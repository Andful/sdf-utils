from typing import Callable
from mdsdf import Sdf2D
from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.workload.computation_node import ComputationNode
from zigzag.workload.Workload import Workload

type StackPolicy = Callable[[Workload, Sdf2D[ComputationNode], Accelerator], list[set[ComputationNode]]]