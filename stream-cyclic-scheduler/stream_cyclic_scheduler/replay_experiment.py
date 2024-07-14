import logging as _logging

_logging_level = _logging.INFO
_logging_format = (
    "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
)
_logging.basicConfig(level=_logging_level, format=_logging_format)

#%%
from zigzag.stages.MainStage import MainStage
from stream.classes.stages.AcceleratorParserStage import (
    AcceleratorParserStage as AcceleratorParserStage_,
)
from stream.classes.stages import UserDefinedModelParserStage, StreamONNXModelParserStage, IntraCoreMappingStage
from stream.visualization.schedule import (
    plot_timeline_brokenaxes, visualize_timeline_plotly
)
from ReplayGenerateCNWorkloadHybridStage import ReplayGenerateCNWorkloadHybridStage;
from ReplayInterCoreMappingStage import ReplayInterCoreMappingStage;
from stream.visualization.memory_usage import plot_memory_usage
from stream.classes.workload.computation_node import ComputationNode
from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.stages import UserDefinedModelParserStage
from zigzag.workload.Workload import Workload
from stream_schedule_solver import Sdf2D
from StackPolicy import StackPolicy

from collections import defaultdict

tiling = defaultdict(lambda : (1, None))

mainstage = MainStage(
    [  # Initializes the MainStage as entry point
        AcceleratorParserStage_,  # Parses the accelerator
        UserDefinedModelParserStage,  # Parses the ONNX Model into the workload
        #StreamONNXModelParserStage,
        ReplayGenerateCNWorkloadHybridStage,
        IntraCoreMappingStage,
        ReplayInterCoreMappingStage,
    ],
    accelerator="architecture/hda_bus.yaml",  # required by AcceleratorParserStage
    workload_path="./workload/mccnn.yaml",  # required by ModelParserStage
    mapping_path="./mapping/mapping.yaml",
    tile_window=dict([
        ("Conv_0", (None, 1)),
        ("Conv_1", (None, 1)),
        ("Conv_2", (None, 1)),
        ("Conv_3", (None, 1)),
        ]),
    schedule_json="schedule.json",
    loma_lpf_limit=2,
    nb_ga_generations = 20,
    nb_ga_individuals = 20,
    plot_hof=True,
    plot_file_name='plot.png',
    operands_to_prefetch = [],
)

[(scme, _), *_] = mainstage.run()

#%%
plot_timeline_brokenaxes(
    scme,
    False,
    section_start_percent=(0,97),
    percent_shown=(3,3),
    plot_data_transfer=True,
    fig_path="/tmp/timeline",
)

plot_memory_usage(scme, (0,97), (3,3), fig_path="/tmp/memory")

visualize_timeline_plotly(scme, fig_path="/tmp/fully_fused_mccnn.html")
# %%
