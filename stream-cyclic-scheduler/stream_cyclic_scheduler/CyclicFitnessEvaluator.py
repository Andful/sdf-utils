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
from stream_schedule_solver import Optimizer
from StackPolicy import StackPolicy
from dataclasses import dataclass
from math import ceil
from typing import Callable, Literal, cast

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

@dataclass(unsafe_hash=True)
class FreeOutput:
    cn: Communication | ComputationNode 
    number_execution: int
    energy: int

    def __init__(self, cn: Communication | ComputationNode):
        self.cn = cn
        self.number_execution = cn.number_execution
        self.energy = 0

Process = Communication | ComputationNode | FreeOutput


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

@dataclass(frozen=True)
class FreeInfo:
    def toJSON(self):
        return {
            "processor-type": "free",
        }

Processor = CoreInfo | ChannelInfo | FreeInfo

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
                if MemoryOperand("I2") in cn.too_large_operands:
                    continue
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

        

    iteration = 0
    def get_stack_fitness(self, sdf: "Sdf2D[ComputationNode]"):
        sdf: "Sdf2D[Process]" = sdf

        from graphviz import Source

        dot = Source(sdf.dot())
        global iteration
        dot.render(f'feed{self.iteration}')
        self.iteration += 1

        memory_sizes = [self.accelerator.get_top_instance_of_core(c, MemoryOperand('I1')).size for c in sorted(self.accelerator.cores.nodes(), key=lambda c: c.id) if c.id != self.accelerator.offchip_core_id]
        def name(p: Process):
            if isinstance(p, ComputationNode):
                return p.name
            elif isinstance(p, Communication):
                return f"{"DRAM" if p.source=="DRAM" else p.source.name}->{"DRAM" if p.target=="DRAM" else p.target.name}"
            elif isinstance(p, FreeOutput):
                return f"FREE({name(p.cn)})"
            else:
                print(type(p))
                assert False
            
        def execution_time(p: Process):
            if isinstance(p, ComputationNode):
                return p.get_runtime()
            elif isinstance(p, Communication):
                return p.latency
            elif isinstance(p, FreeOutput):
                return 0
            
        def padding(p: Process):
            if isinstance(p, ComputationNode):
                if MemoryOperand('I1') in p.constant_operands:
                    return [0, 0]
                node_attr = p.extract_node_attr()
                window = self.tile_window[p.name]
                return [node_attr.padding[LayerDim('IX')][0] if window[0] is not None else 0, node_attr.padding[LayerDim('IY')][1] if window[1] is not None else 0]
            elif isinstance(p, Communication):
                return [0, 0]
            elif isinstance(p, FreeOutput):
                return [0, 0]
        
        def token_size(p: Process):
            if isinstance(p, ComputationNode):
                window = self.tile_window[p.name]
                return p.operand_tensors[LayerOperand('O')].size // ((window[0] or 1)*(window[1] or 1))
            elif isinstance(p, Communication):
                return p.token_size
            elif isinstance(p, FreeOutput):
                return [0, 0]
            
        def kernel_window(p: Process):
            if isinstance(p, ComputationNode):
                window = self.tile_window[p.name]
                if window[0] == None:
                    kernel_x = 1
                else:
                    kernel_x = window[0] + (p.loop_ranges[LayerDim('FX')][1] or 1) - 1
                
                if window[1] == None:
                    kernel_y = 1
                else:
                    kernel_y = window[1] + (p.loop_ranges[LayerDim('FY')][1] or 1) - 1

                return (kernel_x, kernel_y)
            
            elif isinstance(p, Communication):
                if p.source == "DRAM":
                    return kernel_window(p.target)
                else:
                    window = self.tile_window[p.source.name]
                    return (window[0] or 1, window[1] or 1)
            
            elif isinstance(p, FreeOutput):
                return [0, 0]
            
        def memory_id(p: Process):
            if isinstance(p, ComputationNode):
                return p.chosen_core_allocation
            elif isinstance(p, Communication):
                if p.target_core == self.accelerator.offchip_core_id:
                    return None
                return p.target_core
            elif isinstance(p, FreeOutput):
                return 0
            
        def generate_free(p: Process) -> FreeOutput:
            assert isinstance(p, ComputationNode) or isinstance(p, Communication)
            return FreeOutput(p)
            
        processors = [ core for core in  self.accelerator.cores if core.id != self.accelerator.offchip_core_id] + list(set(cl for (_, _, cl) in self.accelerator.cores.edges.data("cl")))
        
        def uses_processor(p: Core | CommunicationLink, task: Process):
            if isinstance(task, FreeOutput):
                return False
            elif isinstance(task, ComputationNode):
                if isinstance(p, Core):
                    return task.chosen_core_allocation == p.id
                elif isinstance(p, CommunicationLink):
                    if len(task.too_large_operands) == 0:
                        return False
                    core = self.accelerator.get_core(task.chosen_core_allocation)
                    offchip = self.accelerator.get_core(self.accelerator.offchip_core_id)
                    if MemoryOperand("O") in task.too_large_operands:
                        links1 = self.accelerator.communication_manager.get_links_for_pair(core, offchip)
                        links2 = self.accelerator.communication_manager.get_links_for_pair(offchip, core)
                        return p in links1 + links2
                    else: #There must be another operator 
                        links = self.accelerator.communication_manager.get_links_for_pair(offchip, core)
                        return p in links
            elif isinstance(task, Communication):
                if isinstance(p, Core):
                    return False
                if task.source != "DRAM" and task.target != "DRAM":
                    core1 = self.accelerator.get_core(task.source.chosen_core_allocation)
                    core2 = self.accelerator.get_core(task.target.chosen_core_allocation)
                    return p in self.accelerator.communication_manager.get_links_for_pair(core1, core2)
                
                offchip = self.accelerator.get_core(self.accelerator.offchip_core_id)
                if task.source == "DRAM":
                    core = self.accelerator.get_core(task.target.chosen_core_allocation)
                    return p in self.accelerator.communication_manager.get_links_for_pair(offchip, core)
                elif task.target == "DRAM":
                    core = self.accelerator.get_core(task.source.chosen_core_allocation)
                    return p in self.accelerator.communication_manager.get_links_for_pair(core, offchip)


            assert False

        from graphviz import Source

        dot = Source(sdf.dot())
        dot.render('test1')

        optimizer = Optimizer(
            optimization_direction=self.optimization_direction,
            memory_sizes=memory_sizes,
            name_fn=name,
            execution_time_fn=execution_time,
            token_size_fn=token_size,
            padding_fn=padding,
            memory_id_fn=memory_id,
            processors=processors,
            uses_processor_fn=uses_processor,
            generate_free_fn=generate_free,
            sdf=sdf
        )

        cycle_time, start_info = optimizer.optimize()

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
                    "layer-operator": "I" if cn.target != "DRAM" and LayerOperand("I") in cn.target.constant_operands else "O",
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
            elif isinstance(cn, FreeOutput):
                p = FreeInfo()
                s = execution.get(p, [])
                execution[p] = s
                s.append({
                    "type": "free",
                    "layer-operator": "O" if isinstance(cn.cn, ComputationNode) or cn.cn.source != "DRAM" else "I",
                    "layer": cn.cn.name if isinstance(cn.cn, ComputationNode) else (cn.cn.source.name if cn.cn.source != "DRAM" else cn.cn.target.name),
                    "core": cn.cn.chosen_core_allocation if isinstance(cn.cn, ComputationNode) else cn.cn.target_core,
                    "repetition-instance": r,
                    "repetition": start_information.repetition(),
                    "after": start_information.start_after(),
                    "start_time": start_information.start_time(),
                    "number_executions": cn.cn.number_execution,
                })
        stack_schedule = StackSchedule(
            raster_direction=self.optimization_direction,
            cycle_time=cycle_time,
            executions=execution,
            layers=[cn.name for cn in sdf.actors() if isinstance(cn, ComputationNode)]
        )

        weight_load_latency, weight_load_energy = self.get_weight_loading_latency_and_energy(sdf)

        execution_latency = 0
        execution_energy = 0

        for ((cn, r), start_information) in start_info:
            start_after: int = start_information.start_after()
            number_execution: int = cn.number_execution
            end_time =  start_information.start_time() + cycle_time*(start_after + number_execution)
            if isinstance(cn, ComputationNode):
                execution_energy += (cn.onchip_energy + cn.offchip_energy)*number_execution
            else:
                execution_energy += cn.energy * number_execution
            execution_latency = max(end_time, execution_latency)

        return (weight_load_energy + execution_energy, weight_load_latency + execution_latency, stack_schedule)

    def get_communication(self, cn1: ComputationNode, from_dram: bool, cn2: ComputationNode, to_dram: bool):
        from_dram = from_dram or MemoryOperand("O") in cn1.too_large_operands
        to_dram = to_dram or MemoryOperand("I1") in cn2.too_large_operands
        if from_dram and to_dram:
            return None
        if not from_dram and not to_dram and cn1.chosen_core_allocation == cn2.chosen_core_allocation:
            return None
        if from_dram:
            core1 = self.accelerator.get_core(self.accelerator.offchip_core_id)
        else: 
            core1 = self.accelerator.get_core(cn1.chosen_core_allocation)

        if to_dram:
            core2 = self.accelerator.get_core(self.accelerator.offchip_core_id)
        else: 
            core2 = self.accelerator.get_core(cn2.chosen_core_allocation)
        
        if cn2 is not None and LayerOperand('I') in cn2.constant_operands:
            tensor = cn2.operand_tensors[LayerOperand('I')]
        else:
            tensor = cn1.operand_tensors[LayerOperand('O')]
        
        energy1 = self.accelerator.get_memory_energy_cost_of_transfer(
                tensor,
                core1,
                core2,
                MemoryOperand('I1'),
                MemoryOperand('I1'),
            )
        links: list[CommunicationLink] = self.accelerator.communication_manager.get_links_for_pair(core1, core2)
        transfer_duration = max([ceil(tensor.size / link.bandwidth) for link in links])
        energy2 = sum(transfer_duration * link.unit_energy_cost for link in links)

        return Communication(
            source="DRAM" if from_dram else cn1,
            target="DRAM" if to_dram else cn2,
            latency=transfer_duration,
            energy=energy1+energy2,
            links=tuple(links),
            token_size=0, # sinking tensor does not use any memory
            target_core=self.accelerator.offchip_core_id if to_dram else cn2.chosen_core_allocation,
            number_execution= cn2.number_execution if cn2 is not None and LayerOperand('I') in cn2.constant_operands else cn1.number_execution
        )

    def get_fitness(self, core_allocations: list[int], return_scme: bool = False):
        """Get the fitness of the given core_allocations

        Args:
            core_allocations (list): core_allocations
        """
        self.set_node_core_allocations(core_allocations)
        stacks = self.stack_policy(self.original_workload, self.sdf, self.accelerator)

        energy = 0
        latency = 0
        stack_schedules: list[StackSchedule] = []

        sources = self.sdf.sources()
        sinks = self.sdf.sinks()
        
        for stack in stacks:
            sdf = self.sdf.induce(set(self.sdf.actors())) # replace with a clone call
            comunications = set()
            for (cn1, cn2) in self.sdf.channels():
                if (cn1 in stack) != (cn2 in stack):
                    c = self.get_communication(cn1, cn1 not in stack, cn2, cn2 not in stack)
                    if c is None:
                        continue
                    comunications.add(c)
                    sdf.cut_channel(
                        channel=(cn1, cn2),
                        new_actor=c
                    )
                elif cn1 in stack:
                    c = self.get_communication(cn1, False, cn2, False)
                    if c is None:
                        continue
                    comunications.add(c)
                    sdf.cut_channel(
                        channel=(cn1, cn2),
                        new_actor=c
                    )
            for cn in stack.intersection(set(sources)):
                c = self.get_communication(None, True, cn, False)
                if c is None:
                    continue
                comunications.add(c)
                sdf.add_actor(c)
                sdf.add_channel(
                    source=c,
                    target=cn,
                    production_rate=(1,),
                    consumption_rate=(1,),
                    initial_tokens=(0,),
                )
            for cn in stack.intersection(set(sinks)):
                c = self.get_communication(cn, False, None, True)
                if c is None:
                    continue
                comunications.add(c)
                sdf.add_actor(c)
                sdf.add_channel(
                    source=cn,
                    target=c,
                    production_rate=(1,),
                    consumption_rate=(1,),
                    initial_tokens=(0,),
                )

            sdf = sdf.induce(stack.union(comunications))
            stack_energy, stack_latency, stack_schedule = self.get_stack_fitness(sdf)
            energy += stack_energy
            latency += stack_latency
            stack_schedules.append(stack_schedule)

        #print(f"energy:{energy}\tlatency:{latency}")
        if not return_scme:
            return energy, latency
        else:
            core_allocation = dict()
            for a in self.sdf.actors():
                a = cast(ComputationNode, a)
                core_allocation[a.name] = a.chosen_core_allocation
            import json
            with open("schedule.json", "w") as f:
                json.dump({
                    "core_allocation": core_allocation,
                    "stack_schedules": list(map(lambda e: e.toJSON(),stack_schedules)),
                    "latency": latency,
                    "energy": energy
                    }, f)
            return energy, latency, None
        
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
            for node in self.sdf.actors():
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
                node.set_chosen_core_allocation(core_allocation)
                node.set_too_large_operands(too_large_operands)
                node.set_offchip_bandwidth(offchip_bw)