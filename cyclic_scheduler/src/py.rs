use mdsdf::{vector::Vector, Channel};
use pyo3::{prelude::*, types::PyDict};

#[derive(Clone, Default)]
struct Task {
    name: String,
    color: String,
    execution_time: usize,
    processor: usize,
}

#[derive(Clone, Default)]
struct Memory {
    cost: f64,
    size: usize,
}

#[derive(Clone, Default)]
#[pyclass(name = "Task")]
struct TaskIndex(usize);

#[derive(Clone, Default)]
#[pyclass(name = "Memory")]
struct MemoryIndex(usize);

#[derive(Clone, Default)]
#[pyclass(name = "RingBuffer")]
struct RingBufferIndex(usize);

#[derive(Clone, Default)]
struct RingBuffer {
    memory: MemoryIndex,
    cost: Vector<2, f64>,
}

#[derive(Clone, Default)]
#[pyclass]
struct CyclicScheduler {
    tasks: Vec<Task>,
    channels: Vec<(Channel<2, isize>, Option<RingBufferIndex>)>,
    ring_buffers: Vec<RingBuffer>,
    memories: Vec<Memory>,
}

#[pymethods]
impl CyclicScheduler {
    #[new]
    fn new() -> Self {
        Default::default()
    }

    fn add_task(
        &mut self,
        name: String,
        execution_time: usize,
        processor: usize,
        color: Option<String>,
    ) -> TaskIndex {
        let result = self.tasks.len();
        self.tasks.push(Task {
            name,
            color: color.unwrap_or("#ffffff".to_string()),
            execution_time,
            processor,
        });
        return TaskIndex(result);
    }

    fn add_dependency(
        &mut self,
        source: TaskIndex,
        target: TaskIndex,
        production_rate: Vector<2, usize>,
        consumption_rate: Vector<2, usize>,
        initial_tokens: Option<Vector<2, isize>>,
        ring_buffer: Option<RingBufferIndex>,
    ) {
        self.channels.push((
            Channel {
                source: source.0,
                target: target.0,
                production_rate: production_rate.into(),
                consumption_rate: consumption_rate.into(),
                initial_tokens: initial_tokens.unwrap_or_default().into(),
            },
            ring_buffer,
        ))
    }

    fn add_ring_buffer(&mut self, cost: Vector<2, f64>, memory: MemoryIndex) -> RingBufferIndex {
        let result = self.ring_buffers.len();
        self.ring_buffers.push(RingBuffer {
            memory,
            cost: cost.into(),
        });
        RingBufferIndex(result)
    }

    fn add_memory(&mut self, cost: f64, memory_size: usize) -> MemoryIndex {
        let result = self.memories.len();
        self.memories.push(Memory {
            cost,
            size: memory_size,
        });
        MemoryIndex(result)
    }

    fn solve(&self) -> CyclicSchedulerSolution {
        use buffer_sizing;
        use grb::prelude::*;
        use std::borrow::Cow;
        let mut sdf = mdsdf::Mdsdf::<2>::new(self.tasks.len());
        for (c, _) in self.channels.iter() {
            sdf.add_channel(c.clone());
        }
        let hsdf = sdf.hsdf();
        let mut milp = milp_formulation::MilpFormulation::new(
            Cow::Borrowed(&hsdf),
            |(i, _)| self.tasks[i].execution_time,
            |(i, _)| self.tasks[i].name.clone(),
        )
        .unwrap();

        let ring_buffer: Vec<Vector<2, Expr>> = (0..self.ring_buffers.len())
            .map({
                let model = &mut milp.model;
                |i| {
                    Ok::<Vector<2, grb::Expr>, grb::Error>(
                        (
                            add_ctsvar!(model, name: &format!("ring_buffer_{i}_0"), bounds: 0..)?
                                + 0.0,
                            add_ctsvar!(model, name: &format!("ring_buffer_{i}_1"), bounds: 0..)?
                                + 0.0,
                        )
                            .into(),
                    )
                }
            })
            .try_collect()
            .unwrap();
        for (i, Memory { size, .. }) in self.memories.iter().enumerate() {
            let allocation = self
                .ring_buffers
                .iter()
                .zip(ring_buffer.iter())
                .filter_map(
                    |(
                        RingBuffer {
                            memory: MemoryIndex(j),
                            cost,
                        },
                        v,
                    )| (*j == i).then(|| *cost * v.clone()),
                )
                .fold(
                    (Expr::from(0.0), Expr::from(0.0)).into(),
                    |a: Vector<2, Expr>, b: Vector<2, Expr>| a + b,
                );
            milp.model
                .add_constr(
                    &format!("memory_allocation_{i}"),
                    c!(allocation[0].clone() + allocation[1].clone() <= size),
                )
                .unwrap();
        }

        let mut buffered_sdf = buffer_sizing::BufferedMrsdf::new(&mut milp);

        for (
            Channel {
                production_rate,
                consumption_rate,
                source,
                target,
                initial_tokens,
            },
            RingBufferIndex(i),
        ) in self
            .channels
            .iter()
            .filter_map(|(a, b)| b.clone().map(|e| (a, e)))
        {
            buffered_sdf
                .add_buffer(Channel {
                    production_rate: *consumption_rate,
                    consumption_rate: *production_rate,
                    source: *target,
                    target: *source,
                    initial_tokens: ring_buffer[i].clone() - initial_tokens.map(|e| e as f64),
                })
                .unwrap()
        }

        crate::cyclic_scheduler(&mut buffered_sdf.milp, |(i, _)| self.tasks[i].processor, 0).unwrap();

        //let buffered_sdf = buffer_sizing::BufferedMrsdf::new(&mut milp);

        buffered_sdf.milp.model.write("solution.lp").unwrap();
        buffered_sdf.milp.model.set_objective(buffered_sdf.milp.throughputs[0], grb::ModelSense::Maximize).unwrap();
        buffered_sdf.milp.model.optimize().unwrap();
        buffered_sdf.milp.model.write("solution.sol").unwrap();
        let throughput = buffered_sdf.milp.model.get_obj_attr(attr::X, &buffered_sdf.milp.throughputs[0]).unwrap();
        CyclicSchedulerSolution{
            throughput,
            tasks: buffered_sdf
                .milp
                .u
                .iter()
                .map(|(a, b)| {
                    let start_time = buffered_sdf.milp.model.get_obj_attr(attr::X, &b).unwrap()/throughput;
                    let Task { name, color, execution_time, processor } = self.tasks[a.0].clone();
                    TaskSolution {
                        start_time,
                        execution_time: execution_time as f64,
                        name,
                        color,
                        processor,
                    }
                }).collect() ,
        }
    }
}

#[derive(Clone, Default)]
#[pyclass(name = "TaskSolution")]
struct TaskSolution {
    start_time: f64,
    execution_time: f64,
    processor: usize,
    name: String,
    color: String,
}

impl ToPyObject for TaskSolution {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let data = PyDict::new_bound(py);
        data.set_item("start_time", self.start_time).unwrap();
        data.set_item("execution_time", self.execution_time).unwrap();
        data.set_item("processor", self.processor).unwrap();
        data.set_item("name", self.name.clone()).unwrap();
        data.set_item("color", self.color.clone()).unwrap();

        data.into()
    }
}

#[pyclass(name = "CyclicSchedulerSolution")]
struct CyclicSchedulerSolution {
    throughput: f64,
    tasks: Vec<TaskSolution>,
}

#[pymethods]
impl CyclicSchedulerSolution {
    fn plot<'py>(&self,  py: Python<'py>) -> PyResult<()> {
        let notebookjs = py.import_bound("notebookjs")?;
        let execute_js = notebookjs.getattr("execute_js")?;

        let data = PyDict::new_bound(py);

        data.set_item("tasks", self.tasks.clone()).unwrap();
        data.set_item("throughput", self.throughput).unwrap();
        execute_js.call1((include_str!("../plotter/dist/bundle.js"), "cyclic_scheduler_plotter.main", data))?;
        Ok(())
    }
}

#[pymodule]
fn cyclic_scheduler(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CyclicScheduler>()?;
    Ok(())
}
