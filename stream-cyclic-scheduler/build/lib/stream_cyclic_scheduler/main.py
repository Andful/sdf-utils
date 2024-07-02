#%%
from zigzag.stages.MainStage import MainStage
from stream.classes.stages.AcceleratorParserStage import (
    AcceleratorParserStage as AcceleratorParserStage_,
)
from stream.classes.stages import StreamONNXModelParserStage, IntraCoreMappingStage, InterCoreMappingStage
from stream.visualization.schedule import (
    plot_timeline_brokenaxes,
)
from MyGenerateCNWorkloadHybridStage import MyGenerateCNWorkloadHybridStage;
from stream.visualization.memory_usage import plot_memory_usage

mainstage = MainStage(
    [  # Initializes the MainStage as entry point
        AcceleratorParserStage_,  # Parses the accelerator
        StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
        MyGenerateCNWorkloadHybridStage,
        IntraCoreMappingStage,
        InterCoreMappingStage,
    ],
    accelerator="architecture/tpu_like_quad_core.yaml",  # required by AcceleratorParserStage
    workload_path="./workload/fsrcnn.onnx",  # required by ModelParserStage
    mapping_path="./mapping/tpu_like_quad_core.yaml",
    cn_define_mode=1,
    hint_loops=[("OX", "all")],
    loma_lpf_limit=2,
    nb_ga_generations = 1,
    nb_ga_individuals = 2,
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

# %%
