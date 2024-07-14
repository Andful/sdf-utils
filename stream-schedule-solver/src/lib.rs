#![feature(iterator_try_collect)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(int_roundings)]

mod no_send;

use std::collections::BTreeMap;
use std::io::Write;
use std::{borrow::Cow, collections::BTreeSet};

use mdsdf::{
    py::{PySdf, Sdf1D, Sdf2D, Sdf3D},
    vector::Vector,
};
use mdsdf::{Channel, ChannelIndex, Mdsdf};
use milp_formulation::{ExecutionTimeT, MilpFormulation, NameT};
use no_send::NoSend;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

struct Name {
    names: Vec<String>,
}

impl NameT<1> for Name {
    fn name(&self, actor: (usize, Vector<1, usize>)) -> String {
        format!("{}({})", self.names[actor.0], actor.1[0])
    }
}

struct ExecutionTime {
    execution_times: Vec<usize>,
}

impl ExecutionTimeT<1> for ExecutionTime {
    fn execution_time(&self, actor: (usize, Vector<1, usize>)) -> usize {
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
    milp: MilpFormulation<'static, 1, ExecutionTime, Name>,
    buffers: BTreeMap<usize, BufferedInformation>,
    memory_occupancy: Vec<grb::Expr>,
    tasks_in_processor: Vec<BTreeSet<usize>>,
}

struct BufferedInformation {
    variable_buffer: grb::Var,
    free_actor: usize,
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
        processors: Vec<Bound<'py, PyAny>>,
        uses_processor_fn: Bound<'py, PyAny>,
        generate_free_fn: Bound<'py, PyAny>,
        pysdf: &Sdf1D,
    ) -> PyResult<(NoSend<Self>, Vec<Py<PyAny>>)> {
        let mut id_map = pysdf
            .0
            .id_map
            .bind(py)
            .iter()
            .map(|(a, b)| PyResult::Ok((b.extract::<usize>()?, a)))
            .try_collect::<Vec<_>>()?;

        id_map.sort_by_key(|(i, _)| *i);

        debug_assert!(id_map.iter().enumerate().all(|(i, (j, _))| i == *j));
        let mut id_map = id_map.into_iter().map(|(_, e)| e).collect::<Vec<_>>();

        let mut sdf: Mdsdf<1> = pysdf.0.sdf.clone();
        let n_actors = sdf.n_actors();
        let channels = sdf.channels.iter().map(Clone::clone).collect::<Vec<_>>();

        let mut free_channels = Vec::new();

        for i in 0..n_actors {
            let free_actor = sdf.add_actor();
            let process = generate_free_fn.call((&id_map[i],), None)?;
            let pr = channels
                .iter()
                .filter(|c| c.source == i)
                .map(|c| c.production_rate)
                .next()
                .unwrap_or([1].into());
            free_channels.push(sdf.add_channel(Channel {
                production_rate: pr,
                consumption_rate: pr,
                source: i,
                target: free_actor,
                initial_tokens: [0].into(),
            }));
            id_map.push(process)
        }

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
        let tasks_in_processor = processors
            .iter()
            .enumerate()
            .map(|(pi, processor)| {
                id_map
                    .iter()
                    .enumerate()
                    .filter_map(|(j, task)| {
                        match uses_processor_fn.call((processor, task), None) {
                            Ok(e) => e,
                            Err(e) => return Some(Err(e)),
                        }
                        .extract::<'_, bool>()
                        .expect("uses_processor_fn should return bool")
                        .then_some(Ok(j))
                    })
                    .try_collect::<BTreeSet<usize>>()
            })
            .try_collect::<Vec<_>>()?;

        let padding = id_map
            .iter()
            .map(|e| padding_fn.call((e,), None)?.extract())
            .try_collect::<Vec<[usize; 2]>>()
            .expect("\"processor\" function must return int");

        debug_assert!(id_map.len() == names.len());
        debug_assert!(id_map.len() == execution_times.len());
        debug_assert!(id_map.len() == padding.len());

        #[derive(Clone)]
        struct BufferInfo {
            core: usize,
            token_size: usize,
            minimum_kernel_size: usize,
            free_actor: usize,
            free_channel: ChannelIndex,
        }

        let mut buffer_info: Vec<Option<BufferInfo>> = (0..n_actors).map(|_| None).collect();
        for channel in channels.iter() {
            let info = &mut buffer_info[channel.source];
            let BufferInfo {
                free_actor,
                minimum_kernel_size,
                ..
            } = if let Some(buffer_info) = info {
                buffer_info
            } else {
                let core: Option<usize> = memory_id_fn
                    .call((&id_map[channel.source],), None)?
                    .extract()?;

                let Some(core) = core else {
                    continue
                };

                let free_actor = channel.source + n_actors;
                id_map.push(generate_free_fn.call((&id_map[channel.source],), None)?);
                let free_channel: ChannelIndex = free_channels[channel.source];

                let token_size = token_size_fn
                    .call((&id_map[channel.source],), None)?
                    .extract()
                    .expect("token_size did not return int");

                *info = Some(BufferInfo {
                    core,
                    token_size,
                    free_actor,
                    free_channel,
                    minimum_kernel_size: 0,
                });
                info.as_mut().unwrap()
            };
            sdf.add_channel(Channel {
                production_rate: channel.consumption_rate,
                consumption_rate: channel.production_rate,
                source: channel.target,
                target: *free_actor,
                initial_tokens: [
                    -(padding[channel.target][optimization_direction.dimension()] as isize),
                ]
                .into(),
            });
        }

        let e = Sdf1D(PySdf {
            id_map: PyDict::from_sequence_bound(
                id_map
                    .iter()
                    .map(Clone::clone)
                    .enumerate()
                    .map(|(i, e)| (e, i))
                    .collect::<Vec<(_, _)>>()
                    .into_py(py)
                    .bind(py),
            )
            .unwrap()
            .unbind(),
            sdf: sdf.clone(),
        });

        let mut f = std::fs::File::create("test.dot").unwrap();
        f.write_fmt(format_args!("{}", e.dot(py).unwrap())).unwrap();
        drop(f);

        let result = NoSend::new({
            use grb::prelude::*;
            move || {
                let mut milp = MilpFormulation::<'static, 1, ExecutionTime, Name>::new(
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

                let buffers = buffer_info
                    .iter()
                    .map(Clone::clone)
                    .enumerate()
                    .filter_map(|(source, info)| {
                        let Some(BufferInfo {
                            core,
                            token_size,
                            free_channel,
                            free_actor,
                            ..
                        }) = info
                        else {
                            return None;
                        };
                        let name = format!("{}_buffer_size", &names[source]);
                        let model = &mut buffered.milp.model;
                        let buffer_size =
                            grb::add_ctsvar!(model, name: &format!("{name}"), bounds: 0..).unwrap();
                        memory_occupancy[core] =
                            memory_occupancy[core].clone() + buffer_size.clone() * token_size;

                        buffered
                            .add_buffer(free_channel, [Expr::from(buffer_size)].into(), &name)
                            .unwrap();

                        Some((
                            source,
                            BufferedInformation {
                                variable_buffer: buffer_size,
                                free_actor,
                            },
                        ))
                    })
                    .collect::<BTreeMap<_, _>>();

                MilpData {
                    milp,
                    optimization_direction,
                    buffers,
                    memory_occupancy,
                    tasks_in_processor,
                }
            }
        });

        Ok((result, id_map.into_iter().map(|e| e.into()).collect()))
    }
}

#[pyclass]
#[derive(Default, Debug, Clone)]
struct ExecutionInformation {
    start_time: usize,
    start_after: isize,
    repetition: usize,
    buffer_size: usize,
}

#[pymethods]
impl ExecutionInformation {
    fn start_time(&self) -> usize {
        self.start_time
    }

    fn start_after(&self) -> isize {
        self.start_after
    }

    fn repetition(&self) -> usize {
        self.repetition
    }

    fn buffer_size(&self) -> usize {
        self.buffer_size
    }
}

#[pyclass]
struct Optimizer {
    id_map: Vec<Py<PyAny>>,
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
        processors: Vec<Bound<'py, PyAny>>,
        uses_processor_fn: Bound<'py, PyAny>,
        generate_free_fn: Bound<'py, PyAny>,
        sdf: &Sdf1D,
    ) -> PyResult<Self> {
        let (milp, id_map) = MilpData::new(
            py,
            optimization_direction,
            memory_sizes,
            name_fn,
            execution_time_fn,
            token_size_fn,
            padding_fn,
            memory_id_fn,
            processors,
            uses_processor_fn,
            generate_free_fn,
            sdf,
        )?;
        Ok(Optimizer { id_map, milp })
    }

    /**
     * Optimization routine
     * Optimize cycle time for raster direction, then other direction, then memory of other direction and then memory of raster direction
     */
    fn optimize<'py>(
        &self,
    ) -> PyResult<(
        usize,
        Vec<((Py<PyAny>, usize), ExecutionInformation)>,
    )> {
        use grb::prelude::*;
        let (throuput, start_information) = self.milp.with(
            move |MilpData {
                      ref mut milp,
                      memory_occupancy,
                      buffers,
                      tasks_in_processor,
                      ..
                  }| {
                cyclic_scheduler::cyclic_scheduler(
                    milp,
                    tasks_in_processor.len(),
                    |(i, _), p| tasks_in_processor[p].contains(&i),
                    0,
                )
                .unwrap();

                milp.model
                    .set_objective(milp.throughputs[0], ModelSense::Maximize)
                    .unwrap();

                milp.model.write("test.lp").unwrap();

                milp.model.optimize().unwrap();

                let throuput = milp
                    .model
                    .get_obj_attr(attr::X, &milp.throughputs[0])
                    .unwrap();

                assert!(throuput > 1e-7); //No deadlock
                assert_eq!(milp.model.status().unwrap(), Status::Optimal);

                milp.model
                    .add_constr("contrain_throuput", c!(milp.throughputs[0] >= throuput))
                    .unwrap();

                let u = milp
                    .u
                    .iter()
                    .map(|(k, v)| {
                        grb::Result::Ok((k.clone(), milp.model.get_obj_attr(attr::X, v)?))
                    })
                    .try_collect::<BTreeMap<_, _>>()
                    .unwrap();

                let model = &mut milp.model;
                let mut memory_sum = Expr::from(0);

                for (i, m) in memory_occupancy.iter().enumerate() {
                    memory_sum = memory_sum + m.clone();
                }

                milp.model
                    .set_objective(memory_sum, ModelSense::Minimize)
                    .unwrap();
                milp.model.optimize().unwrap();

                let mut execution_information = BTreeMap::new();
                for (k, _) in milp.u.iter() {
                    let buffer_size = buffers.get(&k.0);
                    let buffer_size = buffer_size
                        .map(
                            |BufferedInformation {
                                 variable_buffer,
                                 free_actor,
                             }| {
                                milp.model
                                    .get_obj_attr(attr::X, &variable_buffer)
                                    .unwrap()
                                    .floor() as usize
                            },
                        )
                        .unwrap_or(0usize);

                    execution_information.insert(
                        *k,
                        ExecutionInformation {
                            start_time: ((u.get(k).unwrap() % 1.0) / throuput).round() as usize,
                            start_after: u.get(k).unwrap().floor() as isize,
                            repetition: milp.hsdf.repetition_vector[k.0][0],
                            buffer_size,
                        },
                    );
                }

                assert_eq!(milp.model.status().unwrap(), Status::Optimal);

                //model.write("formulation.lp").unwrap();
                //model.write("formulation.sol").unwrap();

                (throuput, execution_information)
            },
        );

        let cycle_time = (1.0 / throuput).round() as usize;
        let mut start_information: Vec<_> = start_information
            .into_iter()
            .map(|((i, j), e)| PyResult::Ok(((self.id_map[i].clone(), j[0]), e)))
            .try_collect()?;
        start_information.sort_by_key(|(_, s)| s.start_time);

        Ok((cycle_time, start_information))
    }
}

#[pymodule]
fn stream_schedule_solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Optimizer>()?;
    m.add_class::<Sdf1D>()?;
    m.add_class::<Sdf2D>()?;
    m.add_class::<Sdf3D>()?;

    Ok(())
}
