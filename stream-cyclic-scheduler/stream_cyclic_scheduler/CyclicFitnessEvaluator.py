from stream.classes.opt.allocation.genetic_algorithm.fitness_evaluator import FitnessEvaluator
from stream.classes.cost_model.cost_model import StreamCostModelEvaluation
from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.hardware.architecture.noc.communication_link import CommunicationLink
from stream.classes.workload.computation_node import ComputationNode
from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.hardware.architecture.Core import Core
from zigzag.workload.Workload import Workload
from zigzag.datatypes import MemoryOperand, LayerOperand, LayerDim
from stream.utils import get_too_large_operands
from mdsdf import Sdf2D
from stream_schedule_solver import Optimizer
from StackPolicy import StackPolicy
from dataclasses import dataclass
from math import ceil
from typing import Callable, Literal

@dataclass(frozen=True)
class Communication:
    source: ComputationNode | Literal["DRAM"]
    target: ComputationNode | Literal["DRAM"]
    latency: int
    energy: int
    links: tuple[CommunicationLink]
    token_size: int
    target_core: int
    number_execution: tuple[int, int]

Process = Communication | ComputationNode


@dataclass(frozen=True)
class CoreInfo:
    id: int
    def toJSON(self):
        return {
            "processor-type": "core",
            "id": self.id,
        }

@dataclass(frozen=True)
class ChannelInfo:
    id: tuple[int | None, int | None]
    def toJSON(self):
        return {
            "processor-type": "channel",
            "source-core": self.id[0],
            "target-core": self.id[1],
        }

Processor = CoreInfo | ChannelInfo

@dataclass
class StackSchedule:
    raster_direction: Literal["X" , "Y"]
    cycle_time: int
    executions: dict[Processor, list[any]]
    layers: list[str]
    def toJSON(self):
        return {
            "raster_direction": self.raster_direction,
            "cycle_time": self.cycle_time,
            "layers": self.layers,
            "execution": list(map(lambda i: {
                "processor": i[0].toJSON(),
                "tasks": i[1]
            }, self.executions.items()))
        }

class CyclicFitnessEvaluator(FitnessEvaluator):
    def __init__(
        self,
        original_workload: Workload,
        accelerator: Accelerator,
        node_hw_performances: dict[ComputationNode, dict[Core, CostModelEvaluation]],
        sdf: "Sdf2D[ComputationNode]",
        stack_policy: StackPolicy,
        tile_window: Callable[[str], tuple[int | None, int | None]],
        optimization_direction: str,
    ) -> None:
        
        self.weights = (-1.0, -1.0)
        self.metrics = ["energy", "latency"]

        self.original_workload = original_workload
        self.accelerator = accelerator
        self.node_hw_performances = node_hw_performances
        self.sdf = sdf
        self.stack_policy = stack_policy
        self.tile_window = tile_window
        self.optimization_direction = optimization_direction
        # self.num_cores = len(inputs.accelerator.cores)

    def get_weight_loading_latency_and_energy(self, sdf: "Sdf2D[ComputationNode]") -> tuple[float, float]:
        dram = self.accelerator.get_core(self.accelerator.offchip_core_id)
        latency = 0
        energy = 0
        for cn in sdf.actors():
            if isinstance(cn, ComputationNode):
                tensor = cn.operand_tensors[LayerOperand('W')]
                destination_core = self.accelerator.get_core(cn.chosen_core_allocation)
                energy1 = self.accelerator.get_memory_energy_cost_of_transfer(
                            tensor,
                            dram,
                            destination_core,
                            MemoryOperand('I1'),
                            MemoryOperand('I1'),
                        )
                
                links: list[CommunicationLink] = self.accelerator.communication_manager.get_links_for_pair(dram, destination_core)
                transfer_duration = max([ceil(tensor.size / link.bandwidth) for link in links])
                energy2 = sum(transfer_duration * link.unit_energy_cost for link in links)

                latency += transfer_duration
                energy += energy1 + energy2

        return (latency, energy)

        


    def get_stack_fitness(self, sdf: "Sdf2D[ComputationNode]"):
        sdf: "Sdf2D[Process]" = sdf
        layers: list[str] = list(map(lambda cn: cn.name, sdf.actors()))
        sources_cn = sdf.sources()
        for (cn1, cn2) in sdf.channels():
            if cn1.chosen_core_allocation != cn2.chosen_core_allocation:
                tensor = cn1.operand_tensors[LayerOperand('O')]
                energy1 = self.accelerator.get_memory_energy_cost_of_transfer(
                        tensor,
                        cn1.chosen_core_allocation,
                        cn2.chosen_core_allocation,
                        MemoryOperand('I1'),
                        MemoryOperand('I1'),
                    )
                core1 = self.accelerator.get_core(cn1.chosen_core_allocation)
                core2 = self.accelerator.get_core(cn2.chosen_core_allocation)
                links: list[CommunicationLink] = self.accelerator.communication_manager.get_links_for_pair(core1, core2)
                transfer_duration = max([ceil(tensor.size / link.bandwidth) for link in links])
                energy2 = sum(transfer_duration * link.unit_energy_cost for link in links)
                window = self.tile_window[cn1.name]
                c = Communication(
                    source=cn1,
                    target=cn2,
                    latency=transfer_duration,
                    energy=energy1+energy2,
                    links=tuple(links),
                    token_size=tensor.size/((window[0] or 1) * (window[1] or 1)), #TODO fix token size
                    target_core=cn2.chosen_core_allocation,
                    number_execution=cn1.number_execution
                )
                sdf.cut_channel((cn1, cn2), c)

        for source_cn in sdf.sources():
            tensor = source_cn.operand_tensors[LayerOperand('I')]
            energy1 = self.accelerator.get_memory_energy_cost_of_transfer(
                    tensor,
                    self.accelerator.offchip_core_id,
                    source_cn.chosen_core_allocation,
                    MemoryOperand('I1'),
                    MemoryOperand('I1'),
                )
            core1 = self.accelerator.get_core(self.accelerator.offchip_core_id)
            core2 = self.accelerator.get_core(source_cn.chosen_core_allocation)
            links: list[CommunicationLink] = self.accelerator.communication_manager.get_links_for_pair(core1, core2)
            transfer_duration = max([ceil(tensor.size / link.bandwidth) for link in links])
            energy2 = sum(transfer_duration * link.unit_energy_cost for link in links)
            c = Communication(
                source="DRAM",
                target=source_cn,
                latency=transfer_duration,
                energy=energy1+energy2,
                links=tuple(links),
                token_size=tensor.size, # Because how Stream works
                target_core=source_cn.chosen_core_allocation,
                number_execution=source_cn.number_execution,
            )

            sdf.add_actor(c)
            sdf.add_channel(c, source_cn, (1, 1), (1, 1), (0, 0))

        for sink_cn in sdf.sinks():
            tensor = sink_cn.operand_tensors[LayerOperand('O')]
            energy1 = self.accelerator.get_memory_energy_cost_of_transfer(
                    tensor,
                    sink_cn.chosen_core_allocation,
                    self.accelerator.offchip_core_id,
                    MemoryOperand('I1'),
                    MemoryOperand('I1'),
                )
            core1 = self.accelerator.get_core(sink_cn.chosen_core_allocation)
            core2 = self.accelerator.get_core(self.accelerator.offchip_core_id)
            links: list[CommunicationLink] = self.accelerator.communication_manager.get_links_for_pair(core1, core2)
            transfer_duration = max([ceil(tensor.size / link.bandwidth) for link in links])
            energy2 = sum(transfer_duration * link.unit_energy_cost for link in links)
            window = self.tile_window[sink_cn.name]
            c = Communication(
                source=sink_cn,
                target="DRAM",
                latency=transfer_duration,
                energy=energy1+energy2,
                links=tuple(links),
                token_size=0, # sinking tensor does not use any memory
                target_core=sink_cn.chosen_core_allocation,
                number_execution=sink_cn.number_execution
            )

            sdf.add_actor(c)
            sdf.add_channel(sink_cn, c, (1, 1), (1, 1), (0, 0))

        

        
        memory_sizes = [self.accelerator.get_top_instance_of_core(c, MemoryOperand('I1')).size for c in sorted(self.accelerator.cores.nodes(), key=lambda c: c.id) if c.id != self.accelerator.offchip_core_id]
        def name(p: Process):
            if isinstance(p, ComputationNode):
                return p.name
            elif isinstance(p, Communication):
                return f"{"DRAM" if p.source=="DRAM" else p.source.name}->{"DRAM" if p.target=="DRAM" else p.target.name}"
            
        def execution_time(p: Process):
            if isinstance(p, ComputationNode):
                return p.get_runtime()
            elif isinstance(p, Communication):
                return p.latency
            
        def padding(p: Process):
            if isinstance(p, ComputationNode):
                if p in sources_cn:
                    return [0, 0]
                node_attr = p.extract_node_attr()
                window = self.tile_window[p.name]
                return [node_attr.padding[LayerDim('IX')][0] if window[0] is not None else 0, node_attr.padding[LayerDim('IY')][1] if window[1] is not None else 0]
            elif isinstance(p, Communication):
                return [0, 0]
        
        def token_sizes(p: Process):
            if isinstance(p, ComputationNode):
                window = self.tile_window[p.name]
                return p.operand_tensors[LayerOperand('O')].size // ((window[0] or 1)*(window[1] or 1))
            elif isinstance(p, Communication):
                return p.token_size
            
        def memory_id(p: Process):
            if isinstance(p, ComputationNode):
                return p.chosen_core_allocation
            elif isinstance(p, Communication):
                return p.target_core
            
        def processor(p: Process):
            if isinstance(p, ComputationNode):
                return p.chosen_core_allocation
            elif isinstance(p, Communication):
                return hash(p.links[0]) & 0xffffffff
        
        from graphviz import Source

        dot = Source(sdf.dot())
        dot.render('test1')

        optimizer = Optimizer(
            self.optimization_direction,
            memory_sizes,
            name,
            execution_time,
            token_sizes,
            padding,
            memory_id,
            processor,
            sdf
        )

        cycle_time, start_info, max_memory = optimizer.optimize()

        execution: dict[Processor, list[Any]] = dict()
        for ((cn, r), start_information) in start_info:
            if isinstance(cn, ComputationNode):
                p = CoreInfo(cn.chosen_core_allocation)
                s = execution.get(p, [])
                execution[p] = s
                s.append({
                    "type": "compute",
                    "layer": cn.name,
                    "repetition-instance": r,
                    "repetition": start_information.repetition(),
                    "after": start_information.start_after(),
                    "start_time": start_information.start_time(),
                    "execution_time": cn.runtime,
                    "energy": cn.onchip_energy + cn.offchip_energy,
                    "number_executions": cn.number_execution,
                    "buffer": start_information.buffer_size(),
                })
            elif isinstance(cn, Communication):
                source = None if cn.links[0].sender == "Any" else cn.links[0].sender.id
                target = None if cn.links[0].receiver == "Any" else cn.links[0].receiver.id
                p = ChannelInfo((source, target))
                s = execution.get(p, [])
                execution[p] = s
                transfer = []
                transfer.append("DRAM" if cn.source == "DRAM" else cn.source.name)
                transfer.append("DRAM" if cn.target == "DRAM" else cn.target.name)

                if cn.source == "DRAM":
                    number_execution = cn.target.number_execution
                else:
                    number_execution = cn.source.number_execution

                s.append({
                    "type": "transfer",
                    "layer-operator": "O" if cn.source != "DRAM" else "I",
                    "layer-source": cn.source.name if cn.source != "DRAM" else None,
                    "layer-target": cn.target.name if cn.target != "DRAM" else None,
                    "repetition-instance": r,
                    "repetition": start_information.repetition(),
                    "after": start_information.start_after(),
                    "start_time": start_information.start_time(),
                    "execution_time": cn.latency,
                    "energy": cn.energy,
                    "number_executions": number_execution,
                    "buffer": start_information.buffer_size(),
                })
        stack_schedule = StackSchedule(
            raster_direction=self.optimization_direction,
            cycle_time=cycle_time,
            executions=execution,
            layers=layers
        )

        weight_load_latency, weight_load_energy = self.get_weight_loading_latency_and_energy(sdf)

        execution_latency = 0
        execution_energy = 0

        for ((cn, r), start_information) in start_info:
            start_after: tuple[int, int] = start_information.start_after()
            number_execution: tuple[int, int] = cn.number_execution
            end_time =  start_information.start_time() + cycle_time*(start_after[0] + number_execution[0])*(start_after[1] + number_execution[1])
            if isinstance(cn, ComputationNode):
                execution_energy += (cn.onchip_energy + cn.offchip_energy)*number_execution[0] * number_execution[1]
            else:
                execution_energy += cn.energy * number_execution[0] * number_execution[1]
            execution_latency = max(end_time, execution_latency)

        return (weight_load_energy + execution_energy, weight_load_latency + execution_latency, stack_schedule)



    def get_fitness(self, core_allocations: list[int], return_scme: bool = False):
        """Get the fitness of the given core_allocations

        Args:
            core_allocations (list): core_allocations
        """
        self.set_node_core_allocations(core_allocations)
        stacks = self.stack_policy(self.original_workload, self.sdf, self.accelerator)
        sdfs = [self.sdf.induce(stack) for stack in stacks]

        energy = 0
        latency = 0
        stack_schedules: list[StackSchedule] = []
        for sdf in sdfs:
            stack_energy, stack_latency, stack_schedule = self.get_stack_fitness(sdf)
            energy += stack_energy
            latency += stack_latency
            stack_schedules.append(stack_schedule)

        import json

        with open("schedule.json", "w") as f:
            json.dump({
                "stack_schedules": list(map(lambda e: e.toJSON(), stack_schedules))
            }, f)

        print(f"energy:{energy}\tlatency:{latency}")
        if not return_scme:
            return energy, latency
        else:
            return energy, latency, None
        raise ValueError("Not implemented")
        scme = StreamCostModelEvaluation(
            self.workload,
            self.accelerator,
            self.operands_to_prefetch,
            self.scheduling_order,
        )
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
            for node in self.sdf.nodes():
                cme = self.node_hw_performances[node][core]
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
                print(f"Setting core allocation to {core_allocation}")
                node.set_chosen_core_allocation(core_allocation)
                node.set_too_large_operands(too_large_operands)
                node.set_offchip_bandwidth(offchip_bw)