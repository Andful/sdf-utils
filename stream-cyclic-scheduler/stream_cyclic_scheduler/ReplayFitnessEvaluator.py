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

        import json

        with open(schedule_json, "r") as f:
            obj = json.load(f)
            self.schedule = obj["stack_schedules"]
            self.core_allocation = obj["core_allocation"]
            for n in self.workload:
                n = cast(ComputationNode, n)
                n.core_allocation = self.core_allocation[n.name]
                n.chosen_core_allocation = n.core_allocation

        self.vm = AcceleratorVirtualMachine(self.accelerator, self.workload)

        #print(self.computation_nodes)

    def get_fitness(self, core_allocations: list[int], return_scme: bool = False):
        """Get the fitness of the given core_allocations

        Args:
            core_allocations (list): core_allocations
        """
        self.set_node_core_allocations(core_allocations)
        scme = StreamCostModelEvaluation(
            self.workload,
            self.accelerator,
            [],
            None,
        )
        computation_nodes: dict[tuple[str, int], ComputationNode] = dict()

        for n in scme.workload.nodes():
            n = cast(ComputationNode, n)
            index = np.array([n.loop_ranges.get(LayerDim('OX'))[0], n.loop_ranges.get(LayerDim('OY'))[0]])
            window = self.tile_window[n.name]
            if window[0] is not None:
                i = index[0]//window[0]
            elif window[1] is not None:
                i = index[1]//window[1]

            computation_nodes[(n.name, i)] = n
        
        self.computation_nodes = computation_nodes
        offchip = self.accelerator.get_core(self.accelerator.offchip_core_id)
        offset = 0
        for stack in self.schedule:
            cycle_time: int = stack["cycle_time"]
            layers: set[str] = set(stack["layers"])
            offset = self.vm.load_weights(layers, offset)
            tasks = []
            for exececution in stack["execution"]:
                tasks += exececution["tasks"]

            tasks = sorted(tasks, key=lambda x: (x["start_time"], 0 if x["type"] == "free" else 1))
            number_of_execution = np.max([np.array(task["after"]) + np.array(task["number_executions"]) for task in tasks], axis=0)
            from tqdm import tqdm
            for i in tqdm(range(number_of_execution)):
                t = cycle_time*i + offset
                for task in tasks:
                    time = t + task["start_time"]
                    after = np.array(task["after"])
                    number_execution = np.array(task["number_executions"])
                    before = after + number_execution
                        
                    if np.all(after <= i) and np.all(i < before):    
                        if task["type"] == "transfer":
                            if task["layer-operator"] == "I":
                                cn = self.computation_nodes[(task["layer-target"], i - after)]
                                core = self.accelerator.get_core(cn.chosen_core_allocation)
                                self.vm.copy_tensor(offchip, core, cn.operand_tensors[LayerOperand("I")], time, task["execution_time"])
                            if task["layer-operator"] == "O":
                                cn1 = self.computation_nodes[(task["layer-source"], i - after)]
                                core1 = self.accelerator.get_core(cn1.chosen_core_allocation)
                                if task["layer-target"] == None:
                                    core2 = offchip
                                else:
                                    cn2 = self.computation_nodes[(task["layer-target"], i - after)]
                                    core2 = self.accelerator.get_core(cn2.chosen_core_allocation) 

                                self.vm.copy_tensor(core1, core2, cn1.operand_tensors[LayerOperand("O")], time, task["execution_time"])
                        elif task["type"] == "compute":
                            cn = self.computation_nodes[(task["layer"], i - after)]
                            print(cn)
                            core = self.accelerator.get_core(cn.chosen_core_allocation)
                            self.vm.compute(core, cn, time, task["execution_time"], task["energy"])
                        elif task["type"] == "free":
                            cn = self.computation_nodes[(task["layer"], i - after)]
                            core = self.accelerator.get_core(task["core"])
                            self.vm.free(core, cn.operand_tensors[LayerOperand(task["layer-operator"])], time)

            offset = t

        energy = self.vm.energy
        latency = self.vm.latency
        scme.energy = self.vm.energy
        scme.latency = self.vm.latency
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
        for node in self.workload:
            core_allocation = self.core_allocation[node.name]
            core = self.accelerator.get_core(core_allocation)
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
