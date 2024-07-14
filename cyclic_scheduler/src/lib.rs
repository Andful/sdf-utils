#![feature(iterator_try_collect)]

use itertools::Itertools;
use mdsdf::vector::Vector;
use milp_formulation::{ExecutionTimeT, MilpFormulation, NameT};

pub fn cyclic_scheduler<const N: usize, ExecutionTime: ExecutionTimeT<N>, Name: NameT<N>>(
    milp: &mut MilpFormulation<'_, N, ExecutionTime, Name>,
    n_processors: usize,
    mut uses_processor: impl FnMut((usize, Vector<N, usize>), usize) -> bool,
    dimension: usize,
) -> grb::Result<()> {
    use grb::prelude::*;
    let processor_assignment = (0..n_processors)
        .map(|p| {
            milp.u
                .keys()
                .map(Clone::clone)
                .filter(|task| uses_processor(*task, p))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let model = &mut milp.model;
    let throughput = milp.throughputs[dimension];
    for tasks in processor_assignment.iter() {
        for (t1, t2) in tasks.iter().tuple_combinations() {
            let task1 = milp.u.get(t1).unwrap();
            let task2 = milp.u.get(t2).unwrap();
            let e1 = milp.execution_time.execution_time(*t1);
            let e2 = milp.execution_time.execution_time(*t2);
            let k = add_intvar!(model, bounds: ..)?;
            model.add_constr(
                "",
                c!(task1.clone() >= task2.clone() + throughput.clone() * e2 - k.clone()),
            )?;
            model.add_constr(
                "",
                c!(task2.clone() >= task1.clone() + throughput.clone() * e1 - (1 - k)),
            )?;
        }
        if tasks.len() == 1 {
            let t = tasks.first().unwrap();
            let task = milp.u.get(t).unwrap();
            let e = milp.execution_time.execution_time(*t);
            let throughput = milp.throughputs[dimension];
            model.add_constr(
                "",
                c!(task.clone() >= task.clone() + throughput.clone() * e - 1),
            )?;
        }
        let cycle_time = tasks
            .iter()
            .map(Clone::clone)
            .map(|actor| milp.execution_time.execution_time(actor))
            .sum::<usize>() as f64;
        model.add_constr("", c!(cycle_time * throughput.clone() <= 1))?;
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use std::borrow::Cow;

    use mdsdf::{vector::Vector, Channel, Mdsdf};
    use milp_formulation::MilpFormulation;

    #[test]
    fn test() {
        let mut sdf = Mdsdf::new(3);
        //Self loops
        sdf.add_channel(Channel {
            production_rate: [1].into(),
            consumption_rate: [1].into(),
            source: 0,
            target: 0,
            initial_tokens: [1].into(),
        });
        sdf.add_channel(Channel {
            production_rate: [1].into(),
            consumption_rate: [1].into(),
            source: 1,
            target: 1,
            initial_tokens: [1].into(),
        });
        sdf.add_channel(Channel {
            production_rate: [1].into(),
            consumption_rate: [1].into(),
            source: 2,
            target: 2,
            initial_tokens: [1].into(),
        });

        sdf.add_channel(Channel {
            production_rate: [2].into(),
            consumption_rate: [3].into(),
            source: 0,
            target: 1,
            initial_tokens: [0].into(),
        });
        sdf.add_channel(Channel {
            production_rate: [1].into(),
            consumption_rate: [2].into(),
            source: 1,
            target: 2,
            initial_tokens: [0].into(),
        });

        //Buffers
        //sdf.add_channel(Channel { production_rate: [3].into(), consumption_rate: [2].into(), source: 1, target: 0, initial_tokens: [5].into() });
        //sdf.add_channel(Channel { production_rate: [3].into(), consumption_rate: [2].into(), source: 1, target: 0, initial_tokens: [5].into() });

        let hsdf = sdf.hsdf();
        struct Name;
        impl NameT<1> for Name {
            fn name(&self, (s, si): (usize, Vector<1, usize>)) -> String {
                let n = ["a", "b", "c"][s];
                format!("{}({})", n, si[0])
            }
        }

        struct ExecutionTime;
        impl ExecutionTimeT<1> for ExecutionTime {
            fn execution_time(&self, (s, _): (usize, Vector<1, usize>)) -> usize {
                [1, 2, 2][s]
            }
        }
        let mut milp = MilpFormulation::new(Cow::Borrowed(&hsdf), ExecutionTime, Name).unwrap();
        cyclic_scheduler(&mut milp, 2,|(i, _), p| [0, 1, 0][i] == p, 0).unwrap();

        let model = &mut milp.model;
        let throughput = &milp.throughputs[0];
        model
            .set_objective(throughput, grb::ModelSense::Maximize)
            .unwrap();
        model.optimize().unwrap();
    }
}
