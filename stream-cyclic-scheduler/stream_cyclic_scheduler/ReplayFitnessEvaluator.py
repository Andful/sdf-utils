from stream.classes.opt.allocation.genetic_algorithm.fitness_evaluator import FitnessEvaluator
from stream.classes.cost_model.cost_model import StreamCostModelEvaluation
from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.workload.computation_node import ComputationNode
from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.datatypes import LayerOperand, LayerDim
from zigzag.hardware.architecture.Core import Core
from zigzag.utils import pickle_deepcopy
import heapq
from stream.utils import get_too_large_operands
from zigzag.workload.Workload import Workload
from AcceleratorVirtualMachine import AcceleratorVirtualMachine
import numpy as np
from typing import cast


class ReplayFitnessEvaluator(FitnessEvaluator):
    """The standard fitness evaluator considers latency, max buffer occupancy and energy equally."""

    def __init__(
        self,
        workload: Workload | None,
        accelerator: Accelerator | None,
        node_hw_performances: dict[ComputationNode, dict[Core, CostModelEvaluation]] | None,
        schedule_json: str,
        tile_window: dict[str, (int | None, int | None)]
    ) -> None:
        super().__init__(workload, accelerator, node_hw_performances)

        self.weights = (-1.0, -1.0)
        self.metrics = ["energy", "latency"]
        self.workload = workload
        self.tile_window = tile_window

        computation_nodes: dict[tuple[str, int, int], ComputationNode] = dict()

        for n in self.workload.nodes():
            n = cast(ComputationNode, n)
            index = np.array([n.loop_ranges.get(LayerDim('OX'))[0], n.loop_ranges.get(LayerDim('OY'))[0]])
            window = self.tile_window[n.name]
            window = np.array([window[0] or index[0], window[1] or index[1]])

            computation_nodes[(n.name, tuple(index//window))] = n
        
        self.computation_nodes = computation_nodes

        import json

        with open(schedule_json, "r") as f:
            self.schedule = json.load(f)

        self.vm = AcceleratorVirtualMachine(self.accelerator, self.workload)

        print(self.computation_nodes)

    def get_fitness(self, core_allocations: list[int], return_scme: bool = False):
        """Get the fitness of the given core_allocations

        Args:
            core_allocations (list): core_allocations
        """
        self.set_node_core_allocations(core_allocations)
        print(self.schedule)
        scme = StreamCostModelEvaluation(
            pickle_deepcopy(self.workload),
            pickle_deepcopy(self.accelerator),
            [],
            None,
        )
        offchip = self.accelerator.get_core(self.accelerator.offchip_core_id)
        offset = 0
        for stack in self.schedule["stack_schedules"]:
            cycle_time: int = stack["cycle_time"]
            layers: set[str] = set(stack["layers"])
            offset = self.vm.load_weights(layers, offset)
            raster_direction = 0 if stack["raster_direction"] == "X" else 1
            other_direction = 1 - raster_direction
            tasks = []
            for exececution in stack["execution"]:
                tasks += exececution["tasks"]

            tasks = sorted(tasks, key=lambda x: x["start_time"])
            number_of_execution = np.max([np.array(task["after"]) + np.array(task["number_executions"]) for task in tasks], axis=0)
            from tqdm import tqdm
            for i in tqdm(range(number_of_execution[other_direction])):
                for j in range(number_of_execution[raster_direction]):
                    t = cycle_time*j + cycle_time*i*number_of_execution[raster_direction]
                    index = np.array([i, j]) if other_direction == 0 else np.array([j, i])
                    for task in tasks:
                        time = t + task["start_time"]
                        print(f"time: {time} i: {i} j:{j}")
                        after = np.array(task["after"])
                        number_execution = np.array(task["number_executions"])
                        before = after + number_execution
                        if np.all(after <= index) and np.all(index < before):
                            index = index*np.array(task["repetition"]) + np.array(task["repetition-instance"])
                            if task["type"] == "transfer":
                                if task["layer-operator"] == "I":
                                    cn = self.computation_nodes[(task["layer-target"], tuple(index))]
                                    core = self.accelerator.get_core(cn.chosen_core_allocation)
                                    self.vm.copy_tensor(offchip, core, cn.operand_tensors[LayerOperand("I")], time, task["execution_time"])
                                    offset += task["execution_time"]
                                if task["layer-operator"] == "O":
                                    print(f"transfering O")
                                    cn1 = self.computation_nodes[(task["layer-source"], tuple(index))]
                                    core1 = self.accelerator.get_core(cn1.chosen_core_allocation)
                                    if task["layer-target"] == None:
                                        core2 = offchip
                                    else:
                                        cn2 = self.computation_nodes[(task["layer-target"], tuple(index))]
                                        core2 = self.accelerator.get_core(cn2.chosen_core_allocation) 

                                    self.vm.copy_tensor(core1, core2, cn1.operand_tensors[LayerOperand("O")], time, task["execution_time"])
                                    offset += task["execution_time"]
                            elif task["type"] == "compute":
                                cn = self.computation_nodes[(task["layer"], tuple(index))]
                                core = self.accelerator.get_core(cn.chosen_core_allocation)
                                self.vm.compute(core, cn, time, task["execution_time"], task["energy"])
            
                

        

        

        energy = scme.energy
        latency = scme.latency
        if not return_scme:
            return energy, latency
        return energy, latency, scme

    def set_node_core_allocations(self, core_allocations: list[int]):
        """Sets the core allocation of all nodes in self.workload according to core_allocations.
        This will only set the energy, runtime and core_allocation of the nodes which are flexible in their core allocation.
        We assume the energy, runtime and core_allocation of the other nodes are already set.

        Args:
            core_allocations (list): list of the node-core allocations
        """
        for i, core_allocation in enumerate(core_allocations):
            core = self.accelerator.get_core(core_allocation)
            (layer_id, group_id) = self.layer_groups_flexible[i]
            # Find all nodes of this coarse id and set their core_allocation, energy and runtime
            nodes = (
                node
                for node in self.workload.nodes()
                if isinstance(node, ComputationNode) and node.id == layer_id and node.group == group_id
            )
            for node in nodes:
                try:
                    equivalent_unique_node = next((n for n in self.node_hw_performances.keys() if node == n))
                except StopIteration:
                    raise ValueError(f"The given node_hw_performances doesn't have run information for node={node}")
                try:
                    cme = self.node_hw_performances[equivalent_unique_node][core]
                except KeyError:
                    raise KeyError(
                        f"The given node_hw_performances doesn't have information for core_allocation={core_allocation} of node={node}"
                    )
                onchip_energy = cme.energy_total  # Initialize on-chip energy as total energy
                latency = cme.latency_total1
                too_large_operands = get_too_large_operands(cme, self.accelerator, core_id=core_allocation)
                # If there is a too_large_operand, we separate the off-chip energy.
                offchip_energy = 0
                for too_large_operand in too_large_operands:
                    layer_operand = next(
                        (k for (k, v) in cme.layer.memory_operand_links.data.items() if v == too_large_operand)
                    )
                    layer_operand_offchip_energy = cme.mem_energy_breakdown[layer_operand][-1]
                    offchip_energy += layer_operand_offchip_energy
                    onchip_energy -= layer_operand_offchip_energy
                # If there was offchip memory added for too_large_operands, get the offchip bandwidth
                offchip_core = self.accelerator.get_core(self.accelerator.offchip_core_id)
                offchip_instance = next(v for k, v in offchip_core.mem_hierarchy_dict.items())[-1].memory_instance
                offchip_bw = cme.get_total_inst_bandwidth(offchip_instance)
                node.set_onchip_energy(onchip_energy)
                node.set_offchip_energy(offchip_energy)
                node.set_runtime(latency)
                node.set_chosen_core_allocation(core_allocation)
                node.set_too_large_operands(too_large_operands)
                node.set_offchip_bandwidth(offchip_bw)
