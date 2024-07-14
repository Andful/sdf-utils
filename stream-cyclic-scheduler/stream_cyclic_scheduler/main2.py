#%%
from zigzag.stages.MainStage import MainStage
from stream.classes.stages.AcceleratorParserStage import (
    AcceleratorParserStage as AcceleratorParserStage_,
)
from stream.classes.stages import UserDefinedModelParserStage, StreamONNXModelParserStage, IntraCoreMappingStage, InterCoreMappingStage
from stream.visualization.schedule import (
    plot_timeline_brokenaxes,
    visualize_timeline_plotly,
)

from TestGenerateCNWorkloadHybridStage import TestGenerateCNWorkloadHybridStage;
from SdfGenerateCNWorkloadHybridStage import SdfGenerateCNWorkloadHybridStage;
from SdfIntraCoreMappingStage import SdfIntraCoreMappingStage;
from SdfInterCoreMappingStage import SdfInterCoreMappingStage;
from stream.visualization.memory_usage import plot_memory_usage
from stream.classes.workload.computation_node import ComputationNode
from stream.classes.hardware.architecture.accelerator import Accelerator
from zigzag.workload.Workload import Workload
from stream_schedule_solver import Sdf2D
from StackPolicy import StackPolicy

def stack_policy(workload: Workload, sdf: "Sdf2D[ComputationNode]", accelerator: Accelerator):
    return [set(sdf.actors())]

stack_policy: StackPolicy = stack_policy

mainstage = MainStage(
    [  # Initializes the MainStage as entry point
        AcceleratorParserStage_,  # Parses the accelerator
        #UserDefinedModelParserStage,  # Parses the ONNX Model into the workload
        StreamONNXModelParserStage,
        TestGenerateCNWorkloadHybridStage,
        IntraCoreMappingStage,
        InterCoreMappingStage,
    ],
    accelerator="architecture/single_core.yaml",  # required by AcceleratorParserStage
    workload_path="./workload/fsrcnn.onnx",  # required by ModelParserStage
    mapping_path="./mapping/single_core.yaml",
    cn_define_mode=1,
    tile_window=dict([
        ("custom_added_Conv1", (1, None)),
        ("custom_added_Conv2", (1, None)),
        ("custom_added_Conv3", (1, None)),
        ("custom_added_Conv4", (1, None)),
        ("custom_added_Conv5", (1, None)),
        ("custom_added_Conv6", (1, None)),
        ("custom_added_Conv7", (1, None)),
        ("custom_added_Conv8", (1, None)),
        ]),
    loma_lpf_limit=2,
    nb_ga_generations = 1,
    nb_ga_individuals = 2,
    plot_hof=True,
    plot_file_name='plot.png',
    operands_to_prefetch = [],
    stack_policy=stack_policy,
)

[(scme, _), *_] = mainstage.run()


print(scme.latency, scme.energy)

"""visualize_timeline_plotly(
    scme,
    draw_communication=True,
    fig_path="/home/andful/Documents/plot.html"
)"""
#%%
plot_timeline_brokenaxes(
    scme,
    False,
    section_start_percent=(0,97),
    percent_shown=(3,3),
    plot_data_transfer=True,
    fig_path="/home/andful/Documents/plot.png",
)

# %%
