#![feature(iterator_try_collect)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(int_roundings)]

mod no_send;

use std::borrow::Cow;
use std::collections::BTreeMap;

use grb::attribute::VarDoubleAttr::Obj;
use mdsdf::{py::PySdf2D, vector::Vector};
use mdsdf::{Channel, ChannelIndex, HsdfChannel, Mdsdf};
use milp_formulation::{ExecutionTimeT, MilpFormulation, NameT};
use no_send::NoSend;
use ordered_float::OrderedFloat;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

struct Name {
    names: Vec<String>,
}

impl NameT<2> for Name {
    fn name(&self, actor: (usize, Vector<2, usize>)) -> String {
        format!("{}({},{})", self.names[actor.0], actor.1[0], actor.1[1])
    }
}

struct ExecutionTime {
    execution_times: Vec<usize>,
}

impl ExecutionTimeT<2> for ExecutionTime {
    fn execution_time(&self, actor: (usize, Vector<2, usize>)) -> usize {
        self.execution_times[actor.0]
    }
}

#[derive(Clone, Copy)]
enum OptimizationDirection {
    X,
    Y,
}

impl OptimizationDirection {
    fn dimension(&self) -> usize {
        match self {
            OptimizationDirection::X => 0,
            OptimizationDirection::Y => 1,
        }
    }

    fn other_dimension(&self) -> usize {
        match self {
            OptimizationDirection::X => 1,
            OptimizationDirection::Y => 0,
        }
    }
}

impl FromPyObject<'_> for OptimizationDirection {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        match ob.extract::<&str>()? {
            "X" => Ok(OptimizationDirection::X),
            "Y" => Ok(OptimizationDirection::Y),
            e => {
                let s =
                    PyString::new_bound(ob.py(), &format!("\"{}\" is not a raster direction", e));
                PyResult::Err(PyErr::from_value_bound(s.into_any()))
            }
        }
    }
}

struct MilpData {
    optimization_direction: OptimizationDirection,
    milp: MilpFormulation<'static, 2, ExecutionTime, Name>,
    buffers: Vec<Option<(grb::Var, usize)>>,
    memory_occupancy: Vec<grb::Expr>,
    processor: Vec<usize>,
}

impl MilpData {
    fn new<'py>(
        py: Python<'py>,
        optimization_direction: OptimizationDirection,
        memory_sizes: Vec<usize>,
        name_fn: Bound<'py, PyAny>,
        execution_time_fn: Bound<'py, PyAny>,
        token_size_fn: Bound<'py, PyAny>,
        padding_fn: Bound<'py, PyAny>,
        memory_id_fn: Bound<'py, PyAny>,
        processor_fn: Bound<'py, PyAny>,
        sdf: &PySdf2D,
    ) -> PyResult<NoSend<Self>> {
        let mut id_map = sdf
            .id_map
            .bind(py)
            .iter()
            .map(|(a, b)| PyResult::Ok((b.extract::<usize>()?, a)))
            .try_collect::<Vec<_>>()?;

        id_map.sort_by_key(|(i, _)| *i);

        debug_assert!(id_map.iter().enumerate().all(|(i, (j, _))| i == *j));

        let id_map = id_map.into_iter().map(|(_, e)| e).collect::<Vec<_>>();
        let names = id_map
            .iter()
            .map(|e| name_fn.call((e,), None)?.extract())
            .try_collect::<Vec<String>>()
            .expect("\"name\" function must return String");
        let execution_times = id_map
            .iter()
            .map(|e| execution_time_fn.call((e,), None)?.extract())
            .try_collect::<Vec<usize>>()
            .expect("\"execution_times\" function must return int");
        let processor = id_map
            .iter()
            .map(|e| processor_fn.call((e,), None)?.extract())
            .try_collect::<Vec<usize>>()
            .expect("\"processor\" function must return int");

        let padding = id_map
            .iter()
            .map(|e| padding_fn.call((e,), None)?.extract())
            .try_collect::<Vec<[usize; 2]>>()
            .expect("\"processor\" function must return int");

        struct BufferInfo {
            core: usize,
            token_size: usize,
            out_channels: Vec<ChannelIndex>,
        }

        let mut buffer_info: Vec<Option<BufferInfo>> =
            (0..sdf.sdf.n_actors).map(|_| None).collect();
        for (i, Channel { source, .. }) in sdf.sdf.channels.iter().enumerate() {
            let info = &mut buffer_info[*source];
            if let Some(BufferInfo { out_channels, .. }) = info {
                out_channels.push(ChannelIndex(i));
            } else {
                let core = memory_id_fn.call((&id_map[*source],), None)?.extract()?;

                let token_size = token_size_fn
                    .call((&id_map[*source],), None)?
                    .extract()
                    .expect("token_size did not return int");

                *info = Some(BufferInfo {
                    core,
                    token_size,
                    out_channels: vec![ChannelIndex(i)],
                });
            }
        }

        let result = NoSend::new({
            use grb::prelude::*;
            let sdf: Mdsdf<2> = sdf.sdf.clone();
            move || {
                let mut milp = MilpFormulation::<'static, 2, ExecutionTime, Name>::new(
                    Cow::Owned(sdf.into_hsdf()),
                    ExecutionTime { execution_times },
                    Name {
                        names: names.clone(),
                    },
                )
                .unwrap();
                let mut buffered = buffer_sizing::BufferedMrsdf::new(&mut milp);
                let mut memory_occupancy = (0..memory_sizes.len())
                    .map(|_| grb::Expr::from(0))
                    .collect::<Vec<_>>();

                let channels = buffered.milp.hsdf.mdsdf.channels.clone();
                let buffers = buffer_info
                    .into_iter()
                    .enumerate()
                    .map(|(source, info)| {
                        let Some(BufferInfo {
                            core,
                            token_size,
                            out_channels,
                        }) = info
                        else {
                            return None;
                        };

                        let minimum_buffer = out_channels
                            .iter()
                            .map(|ChannelIndex(i)| {
                                let Channel {
                                    production_rate,
                                    consumption_rate,
                                    initial_tokens,
                                    target,
                                    ..
                                } = channels[*i];
                                let production_rate = production_rate
                                    [optimization_direction.other_dimension()]
                                    as isize;
                                let consumption_rate = consumption_rate
                                    [optimization_direction.other_dimension()]
                                    as isize;
                                let initial_tokens = initial_tokens
                                    [optimization_direction.other_dimension()]
                                    as isize;
                                let divisor = num::integer::gcd(production_rate, consumption_rate);
                                consumption_rate - divisor + production_rate
                                    - (initial_tokens.div_floor(production_rate)) * production_rate
                                    + padding[target][optimization_direction.other_dimension()] as isize
                            })
                            .chain(std::iter::once(0))
                            .max()
                            .expect("This should not be none")
                            as usize;
                        println!("minimum_buffer {} {}", &names[source], minimum_buffer);
                        let name = format!("{}_buffer_size", &names[source]);
                        let model = &mut buffered.milp.model;
                        let buffer_size =
                            grb::add_ctsvar!(model, name: &format!("{name}"), bounds: 0..).unwrap();
                        memory_occupancy[core] = memory_occupancy[core].clone()
                            + buffer_size.clone() * token_size * minimum_buffer;

                        for channel_index in out_channels {
                            let target = buffered.milp.hsdf.mdsdf.channels[channel_index.0].target;

                            buffered
                                .add_buffer(
                                    channel_index,
                                    match optimization_direction {
                                        OptimizationDirection::X => {
                                            [Expr::from(buffer_size - padding[target][0]), (minimum_buffer - padding[target][1]).into()]
                                        }
                                        OptimizationDirection::Y => {
                                            [(minimum_buffer - padding[target][0]).into(), (buffer_size - padding[target][1]).into()]
                                        }
                                    }
                                    .into(),
                                    &name,
                                )
                                .unwrap();
                        }

                        Some((buffer_size, minimum_buffer))
                    })
                    .collect::<Vec<_>>();

                MilpData {
                    milp,
                    optimization_direction,
                    buffers,
                    memory_occupancy,
                    processor,
                }
            }
        });

        Ok(result)
    }
}

#[pyclass]
#[derive(Default, Debug, Clone)]
struct ExecutionInformation {
    start_time: usize,
    start_after: [isize; 2],
    repetition: [usize; 2],
    buffer_size: [usize; 2],
}

#[pymethods]
impl ExecutionInformation {
    fn start_time(&self) -> usize {
        self.start_time
    }

    fn start_after(&self) -> (isize, isize) {
        (self.start_after[0], self.start_after[1])
    }

    fn repetition(&self) -> (usize, usize) {
        (self.repetition[0], self.repetition[1])
    }

    fn buffer_size(&self) -> (usize, usize) {
        (self.buffer_size[0], self.buffer_size[1])
    }
}

#[pyclass]
struct Optimizer {
    id_map: Py<PyDict>,
    milp: NoSend<MilpData>,
}

#[pymethods]
impl Optimizer {
    #[new]
    fn new<'py>(
        py: Python<'py>,
        optimization_direction: OptimizationDirection,
        memory_sizes: Vec<usize>,
        name_fn: Bound<'py, PyAny>,
        execution_time_fn: Bound<'py, PyAny>,
        token_size_fn: Bound<'py, PyAny>,
        padding_fn: Bound<'py, PyAny>,
        memory_id_fn: Bound<'py, PyAny>,
        processor_fn: Bound<'py, PyAny>,
        sdf: &PySdf2D,
    ) -> PyResult<Self> {
        Ok(Optimizer {
            id_map: sdf.id_map.clone(),
            milp: MilpData::new(
                py,
                optimization_direction,
                memory_sizes,
                name_fn,
                execution_time_fn,
                token_size_fn,
                padding_fn,
                memory_id_fn,
                processor_fn,
                sdf,
            )?,
        })
    }

    /**
     * Optimization routine
     * Optimize cycle time for raster direction, then other direction, then memory of other direction and then memory of raster direction
     */
    fn optimize<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        usize,
        Vec<((Bound<'py, PyAny>, (usize, usize)), ExecutionInformation)>,
        usize,
    )> {
        use grb::prelude::*;
        let (throuput, start_information, memory) = self.milp.with(
            move |MilpData {
                      ref mut milp,
                      optimization_direction,
                      memory_occupancy,
                      buffers,
                      processor,
                      ..
                  }| {

                cyclic_scheduler::cyclic_scheduler(
                    milp,
                    |(i, _)| processor[i],
                    optimization_direction.dimension(),
                )
                .unwrap();

                milp.model
                    .set_objective(
                        milp.throughputs[optimization_direction.dimension()],
                        ModelSense::Maximize,
                    )
                    .unwrap();
                
                milp.model.write("f.lp").unwrap();

                milp.model.optimize().unwrap();

                let throuput1 = milp
                    .model
                    .get_obj_attr(
                        attr::X,
                        &milp.throughputs[optimization_direction.dimension()],
                    )
                    .unwrap();

                assert!(throuput1 > 1e-7); //No deadlock
                assert_eq!(milp.model.status().unwrap(), Status::Optimal);

                milp.model
                    .add_constr(
                        "t",
                        c!(milp.throughputs[optimization_direction.dimension()] >= throuput1),
                    )
                    .unwrap();

                milp.model
                    .set_objective(
                        milp.throughputs[optimization_direction.other_dimension()],
                        ModelSense::Maximize,
                    )
                    .unwrap();
                milp.model.optimize().unwrap();

                let throuput2 = milp
                    .model
                    .get_obj_attr(
                        attr::X,
                        &milp.throughputs[optimization_direction.other_dimension()],
                    )
                    .unwrap();

                assert!(throuput2 > 1e-7); //No deadlock
                assert_eq!(milp.model.status().unwrap(), Status::Optimal);

                milp.model
                    .add_constr(
                        "t",
                        c!(milp.throughputs[optimization_direction.other_dimension()] >= throuput2),
                    )
                    .unwrap();

                let model = &mut milp.model;
                let max_memory = add_ctsvar!(model, name: "max_memory", bounds: 0..).unwrap();

                for (i, m) in memory_occupancy.iter().enumerate() {
                    milp.model
                        .add_constr(&format!("max_memory_{i}"), c!(max_memory >= m.clone()))
                        .unwrap();
                }

                milp.model
                    .set_objective(max_memory, ModelSense::Minimize)
                    .unwrap();
                milp.model.optimize().unwrap();

                milp.model.write("f.sol").unwrap();

                let mut execution_information = BTreeMap::new();
                for ((k, v), buffer_size) in milp.u.iter().zip(buffers) {
                    let buffer_size = buffer_size
                        .map(|(variable_size, minimum_size)| {
                            (
                                milp.model
                                    .get_obj_attr(attr::X, &variable_size)
                                    .unwrap()
                                    .floor() as usize,
                                minimum_size,
                            )
                        })
                        .map(
                            |(variable_size, minimum_size)| match optimization_direction {
                                OptimizationDirection::X => [variable_size, minimum_size],
                                OptimizationDirection::Y => [minimum_size, variable_size],
                            },
                        )
                        .unwrap_or([0, 0]);

                    let u = milp.model.get_obj_attr(attr::X, v).unwrap();
                    execution_information.insert(*k, ExecutionInformation {
                        start_time: ((u % 1.0) * throuput1).round() as usize,
                        start_after: match optimization_direction {
                            OptimizationDirection::X => [u.floor() as isize, u.floor() as isize],
                            OptimizationDirection::Y => [u.floor() as isize, u.floor() as isize],
                        },
                        repetition: [
                            milp.hsdf.repetition_vector[k.0][0],
                            milp.hsdf.repetition_vector[k.0][1]
                        ],
                        buffer_size
                    });
                }

                assert_eq!(milp.model.status().unwrap(), Status::Optimal);

                let memory = milp.model.get_obj_attr(attr::X, &max_memory).unwrap();
                milp.model
                    .add_constr("memory_constraint", c!(max_memory >= memory))
                    .unwrap();

                //model.write("formulation.lp").unwrap();
                //model.write("formulation.sol").unwrap();

                (throuput1, execution_information, memory)
            },
        );
        let mut id_map = self
            .id_map
            .bind(py)
            .iter()
            .map(|(a, b)| PyResult::Ok((b.extract()?, a)))
            .try_collect::<Vec<(usize, Bound<'py, PyAny>)>>()?;
        id_map.sort_by_key(|(i, _)| *i);
        let id_map = id_map.into_iter().map(|(_, e)| e).collect::<Vec<_>>();

        let cycle_time = (1.0 / throuput).round() as usize;
        let mut start_information: Vec<_> = start_information
            .into_iter()
            .map(|((i, j), e)| PyResult::Ok(((id_map[i].clone(), (j[0], j[1])), e)))
            .try_collect()?;
        start_information.sort_by_key(|(_, s)| s.start_time);

        println!("Start Information: {:?}", start_information);

        Ok((cycle_time, start_information, memory.round() as usize))
    }
}

#[pymodule]
fn stream_schedule_solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Optimizer>()?;
    m.add_class::<PySdf2D>()?;

    Ok(())
}
