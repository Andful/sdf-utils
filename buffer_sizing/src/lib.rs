#![feature(iterator_try_collect)]

use mdsdf::{util::bounded_iterator, vector::Vector, Channel};
use milp_formulation::{MilpFormulation, ExecutionTimeT, NameT};
use std::{borrow::Cow, collections::BTreeMap, isize};

pub struct BufferedMrsdf<'a, const N: usize, ExecutionTime: ExecutionTimeT<N>, Name: NameT<N>> {
    pub milp: &'a mut MilpFormulation<'a, N, ExecutionTime, Name>,
}

impl<'a, const N: usize, ExecutionTime: ExecutionTimeT<N>, Name: NameT<N>> BufferedMrsdf<'a, N, ExecutionTime, Name> {
    pub fn new(
        milp: &'a mut MilpFormulation<'a, N, ExecutionTime, Name>
    ) -> Self {
        Self {
            milp,
        }
    }

    pub fn add_buffer(&mut self, Channel { production_rate, consumption_rate, source, target, initial_tokens }: Channel<N, grb::Expr>) -> grb::Result<()> {
        use grb::prelude::*;

        assert!(self.milp.hsdf.repetition_vector[source]*production_rate == self.milp.hsdf.repetition_vector[target]*consumption_rate, "Repetition vector would change");
        let model = &mut self.milp.model;
        let denominator = self.milp.hsdf.repetition_vector[source]*production_rate;
        let one_over_denominator = Vector::from([1.0;N])/denominator.map(|e| e as f64);

        let mut memoized: [BTreeMap::<usize, grb::Var>;N] = (0..N).map(|_| Default::default()).collect::<Vec<_>>().try_into().unwrap();
        for index in bounded_iterator(Cow::Borrowed(&denominator)) {
            for (d, tokens) in index.iter().enumerate() {
                let v = add_intvar!(model, bounds: 0..)?;
                memoized[d].insert(*tokens, v.clone());
                let den = denominator[d];
                let name = format!("(d+{tokens})/{den}");
                let tokens = *tokens as f64;
                model.add_constr(&name, c!(v <= (initial_tokens[d].clone() + tokens)*one_over_denominator[d]))?;
            }
        }

        for si in bounded_iterator(Cow::Borrowed(&self.milp.hsdf.repetition_vector[source])) {
            for ti in bounded_iterator(Cow::Borrowed(&self.milp.hsdf.repetition_vector[target])) {
                let tokens = (production_rate*si).map(|e| e as isize) - (consumption_rate*ti).map(|e| e as isize) + (denominator.map(|e| e as isize) - consumption_rate.map(|e| e as isize));
                let to_floor_tokens = tokens.modulo(denominator.map(|e| e as isize)).map(|e| e as usize);
                let hsdf_tokens = tokens.flooring_div(denominator.map(|e| e as isize));
                //let MilpFormulation { u, throughputs, model, execution_time, .. } = &mut self.milp_formulation;
                let u = &self.milp.u;
                let throughputs = &self.milp.throughputs;
                let execution_time = &mut self.milp.execution_time;
                let u_source = u.get(&(source, si)).unwrap().clone();
                let u_target = u.get(&(target, ti)).unwrap().clone();
                for d in 0..N {
                    let tokens = memoized[d].get(&to_floor_tokens[d]).unwrap().clone() + hsdf_tokens[d];
                    let throuput = throughputs[d];
                    let et = execution_time((source, si));
                    model.add_constr("", c!(u_target >= u_source + et*throuput - tokens))?;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use mdsdf::Mdsdf;
    use super::*;
    use grb::prelude::*;
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
        let execution_time = Box::new(move |(s, _)| *execution_times.get(&s).unwrap());
        let name = Box::new(move |(s,si): (usize, Vector<1, usize>)| format!("{}({})",names.get(&s).unwrap().to_string(), si[0]));
        let mut milp = MilpFormulation::new(Cow::Borrowed(&hsdf), execution_time, name).unwrap();
        let mut buffered = BufferedMrsdf::new(&mut milp);

        let model = &mut buffered.milp.model;
        let buffer1 = add_ctsvar!(model, name: "buffer1", bounds: 0..4).unwrap();
        let buffer2 = add_ctsvar!(model, name: "buffer2", bounds: 0..3).unwrap();
        buffered.add_buffer(Channel { production_rate: [3].into(), consumption_rate: [2].into(), source: 1, target: 0, initial_tokens: [buffer1.into()].into() }).unwrap();
        buffered.add_buffer(Channel { production_rate: [2].into(), consumption_rate: [1].into(), source: 2, target: 1, initial_tokens: [buffer2.into()].into() }).unwrap();
        
        let model = &mut buffered.milp.model;
        let throughput = &buffered.milp.throughputs[0];
        model.set_objective(throughput, grb::ModelSense::Maximize).unwrap();
        model.optimize().unwrap();
    }
}
