#![feature(iterator_try_collect)]

mod py;

use std::collections::BTreeMap;
use mdsdf::vector::Vector;
use milp_formulation::{MilpFormulation, ExecutionTimeT, NameT};
use itertools::Itertools;

pub fn cyclic_scheduler<const N: usize, ExecutionTime: ExecutionTimeT<N>, Name: NameT<N>>(milp: &mut MilpFormulation<'_, N, ExecutionTime, Name>, mut processor: impl FnMut((usize,Vector<N, usize>)) -> usize, dimension: usize) -> grb::Result<()> {
    use grb::prelude::*;
    let mut processor_assignment: BTreeMap<usize, Vec<(usize,Vector<N, usize>)>> = Default::default();
    for k in milp.u.keys() {
        processor_assignment.entry(processor(*k)).or_default().push(*k);
    }

    let model = &mut milp.model;
    let throughput = milp.throughputs[dimension];
    for tasks in processor_assignment.values() {

        for (t1, t2) in tasks.iter().tuple_combinations() {
            let task1 = milp.u.get(t1).unwrap();
            let task2 = milp.u.get(t2).unwrap();
            let e1 = (milp.execution_time)(*t1);
            let e2 = (milp.execution_time)(*t2);
            let k = add_intvar!(model, bounds: ..)?;
            model.add_constr("", c!(task1.clone() >= task2.clone() + throughput.clone()*e2 - k.clone()))?;
            model.add_constr("", c!(task2.clone() >= task1.clone() + throughput.clone()*e1 - (1 - k)))?;
        }
        if tasks.len() == 1 {
            let t = tasks.first().unwrap();
            let task = milp.u.get(t).unwrap();
            let e = (milp.execution_time)(*t);
            let throughput = milp.throughputs[dimension];
            model.add_constr("", c!(task.clone() >= task.clone() + throughput.clone()*e - 1))?;
        }
        let cycle_time = tasks.iter().map(Clone::clone).map(&mut milp.execution_time).sum::<usize>() as f64;
        model.add_constr("", c!(cycle_time * throughput.clone() <= 1))?;
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use std::{borrow::Cow, collections::BTreeMap};

    use mdsdf::{vector::Vector, Channel, Mdsdf};
    use milp_formulation::MilpFormulation;

    #[test]
    fn test() {
        let names: BTreeMap<usize, &str> = ["a", "b", "c"].iter().map(|e| *e).enumerate().collect();
        let execution_times: BTreeMap<usize, usize> = [1, 2, 2].iter().map(Clone::clone).enumerate().collect();
        let mut sdf = Mdsdf::new(3);
        //Self loops
        sdf.add_channel(Channel { production_rate: [1].into(), consumption_rate: [1].into(), source: 0, target: 0, initial_tokens: [1].into() });
        sdf.add_channel(Channel { production_rate: [1].into(), consumption_rate: [1].into(), source: 1, target: 1, initial_tokens: [1].into() });
        sdf.add_channel(Channel { production_rate: [1].into(), consumption_rate: [1].into(), source: 2, target: 2, initial_tokens: [1].into() });

        sdf.add_channel(Channel { production_rate: [2].into(), consumption_rate: [3].into(), source: 0, target: 1, initial_tokens: [0].into() });
        sdf.add_channel(Channel { production_rate: [1].into(), consumption_rate: [2].into(), source: 1, target: 2, initial_tokens: [0].into() });

        //Buffers
        //sdf.add_channel(Channel { production_rate: [3].into(), consumption_rate: [2].into(), source: 1, target: 0, initial_tokens: [5].into() });
        //sdf.add_channel(Channel { production_rate: [3].into(), consumption_rate: [2].into(), source: 1, target: 0, initial_tokens: [5].into() });

        let hsdf = sdf.hsdf();
        let execution_time = Box::new({let execution_times = execution_times.clone(); move |(s, _)| *execution_times.get(&s).unwrap()});
        let name = Box::new(move |(s,si): (usize, Vector<1, usize>)| format!("{}({})",names.get(&s).unwrap().to_string(), si[0]));
        let mut milp = MilpFormulation::new(Cow::Borrowed(&hsdf), execution_time, name).unwrap();
        cyclic_scheduler(&mut milp,|(i, _)| [0, 1, 0][i],0).unwrap();

        let model = &mut milp.model;
        let throughput = &milp.throughputs[0];
        model.set_objective(throughput, grb::ModelSense::Maximize).unwrap();
        model.optimize().unwrap();
    }
}
