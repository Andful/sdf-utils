from stream.classes.hardware.architecture.accelerator import Accelerator, Core
from stream.classes.hardware.architecture.noc.communication_link import CommunicationLink
from stream.classes.cost_model.communication_manager import CommunicationLinkEvent
from stream.classes.workload.tensor import Tensor
from stream.classes.workload.computation_node import ComputationNode
from zigzag.datatypes import LayerDim, LayerOperand
from zigzag.workload.Workload import Workload
from logging import info, warning
from typing import Dict, cast
from math import ceil

class AcceleratorVirtualMachine:
    accelerator: Accelerator
    latency: int
    energy: int

    def __init__(self, accelerator: Accelerator, workload: Workload):
        self.latency = 0
        self.energy = 0
        self.loaded_weights: set[tuple[int, Tensor]] = set()
        from stream.classes.cost_model.scheduler import initialize_offchip_tensors
        self.accelerator = accelerator
        self.workload = workload
        initialize_offchip_tensors(workload, accelerator)

    def free_weights(self, start: int):
        for (core_id, tensor) in self.loaded_weights:
            core = self.accelerator.get_core(core_id)
            self.free(core, tensor, start)

    def load_weights(self, stack_layers: set[str], start: int) -> int:
        offchip_core_id = self.accelerator.offchip_core_id
        offchip_core = self.accelerator.get_core(offchip_core_id)

        self.loaded_weights = set()
        for n in self.workload.nodes():
            n = cast(ComputationNode, n)
            if n.name not in stack_layers:
                continue
            for op, tensor in n.operand_tensors.items():
                if tensor.layer_operand == LayerOperand("W"):
                    # Move tensor
                    core_id = n.chosen_core_allocation
                    if (core_id, tensor) in self.loaded_weights:
                        continue
                    self.loaded_weights.add((core_id, tensor))
                    core = self.accelerator.get_core(n.chosen_core_allocation)
                    [communication_link, *
                        _] = self.accelerator.communication_manager.get_links_for_pair(offchip_core, core)
                    transfer_time = ceil(
                        tensor.size/communication_link.bandwidth)
                    self.copy_tensor(offchip_core, core, tensor,
                                     start, transfer_time)
                    start += transfer_time
        return start

    def copy_tensor(self, core1: Core, core2: Core, tensor: Tensor, start: int, duration: int):
        info(f"Copying {tensor} at {start} from {core1} to {core2}")
        links: list[CommunicationLink] = self.accelerator.communication_manager.get_links_for_pair(
            core1, core2)
        link = max(links, key=lambda link: ceil(tensor.size / link.bandwidth))
        duration2 = ceil(tensor.size / link.bandwidth)
        assert duration == duration2
        energy = link.unit_energy_cost*duration
        link.transfer(
            CommunicationLinkEvent(
                "transfer",
                start=start,
                end=start + duration,
                tensors=[tensor],
                energy=energy
            )
        )
        self.energy += energy

        self.energy += self.accelerator.get_memory_energy_cost_of_transfer(
            tensor, core1, core2, tensor.memory_operand, tensor.memory_operand
        )

        self.accelerator.memory_manager.add_tensor_to_core(
            tensor=tensor,
            core=core2,
            timestep=start,
            timestep_end=start + duration,
        )

        memory_op = tensor.memory_operand
        top_instance = self.accelerator.get_top_instance_of_core(
            core1, memory_op)
        if not self.accelerator.memory_manager.contains(tensor, top_instance):
            warning(f"no {tensor} in {core1} {top_instance}")

        self.latency = max(self.latency, start + duration)

    def free(self, core: Core, tensor: Tensor, start: int):
        info(f"Freeing {tensor} at {start} in {core}")
        memory_op = tensor.memory_operand
        top_instance = self.accelerator.get_top_instance_of_core(
            core, memory_op)
        if not self.accelerator.memory_manager.contains(tensor, top_instance):
            info(f"WARNING: no {tensor} in {core} {top_instance}")
        self.accelerator.memory_manager.remove_tensor_from_top_instance(
            top_instance,
            tensor,
            start,
        )

    def compute(self, core: Core, cn: ComputationNode, start: int, duration: int, energy: int):
        info(f"Executing {cn.name} at {start} producing {
             cn.operand_tensors[LayerOperand('O')]} in {core}")
        out_tensor = cn.operand_tensors[LayerOperand('O')]
        print(f"computed {out_tensor}")

        self.accelerator.memory_manager.add_tensor_to_core(
            tensor=out_tensor,
            core=core,
            timestep=start,
            timestep_end=start + duration,
        )

        cn.chosen_core_allocation = core.id
        cn.start = start
        cn.runtime = duration
        self.energy += energy

        # if cn.onchip_energy is not None:
        # print(cn.onchip_energy, energy, type(cn.onchip_energy))
        # assert cn.onchip_energy == energy # Not sure what is happening here
