#![feature(iterator_try_collect)]
#![feature(trait_alias)]

use mdsdf::{vector::Vector, Hsdf, HsdfChannel};
use std::{borrow::Cow, collections::BTreeMap};

pub trait ExecutionTimeT<const N: usize> = FnMut((usize, Vector<N, usize>)) -> usize;

pub trait NameT<const N: usize> = FnMut((usize, Vector<N, usize>)) -> String;

pub struct MilpFormulation<'a, const N: usize, ExecutionTime: ExecutionTimeT<N>, Name: NameT<N>> {
    pub hsdf: Cow<'a, Hsdf<'a, N>>,
    pub model: grb::Model,
    pub u: BTreeMap<(usize, Vector<N, usize>), grb::Var>,
    pub throughputs: Vec<grb::Var>,
    pub execution_time: ExecutionTime,
    pub name: Name,
}

impl<'a, const N: usize, ExecutionTime: ExecutionTimeT<N>, Name: NameT<N>> MilpFormulation<'a, N, ExecutionTime, Name> {
    pub fn new(
        hsdf: Cow<'a, Hsdf<'a, N>>,
        mut execution_time: ExecutionTime,
        mut name: Name,
    ) -> grb::Result<Self> {
        use grb::prelude::*;

        let mut model = Model::new("model")?;

        let throughputs = (0..N)
            .map(|i| add_ctsvar!(model, name: &format!("throughput_{i}"), bounds: 0.0..))
            .try_collect::<Vec<_>>()?;

        let u: BTreeMap<(usize, Vector<N, usize>), Var> = hsdf
            .actors()
            .map(|a| add_ctsvar!(model, name: &name(a), bounds: 0.0..).map(|v| (a, v)))
            .try_collect()?;

        for HsdfChannel {
            source,
            target,
            initial_tokens,
        } in hsdf.channels()
        {
            let u_source = u.get(&source).unwrap();
            let u_target = u.get(&target).unwrap();
            for i in 0..N {
                let u_source = u_source.clone();
                let u_target = u_target.clone();
                let e = execution_time(source);
                let throughput = throughputs[i].clone();
                let initial_tokens = initial_tokens[i] as f64;
                model.add_constr(
                    &format!("dependency_{}_{}", name(source), name(target)),
                    c!(u_target >= u_source + e * throughput - initial_tokens),
                )?;
            }
        }

        Ok(Self {
            hsdf,
            model,
            u,
            throughputs,
            name,
            execution_time
        })
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use mdsdf::{Channel, Mdsdf};

    #[test]
    fn test() {
        //Exa,ple from "Exploring Trade≠Offs in Buffer Requirements and Throughput Constraints for Synchronous Datafow Graphs"
        let mut sdf = Mdsdf::<1>::new(3);
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
            source: 1,
            target: 1,
            initial_tokens: [1].into(),
        });

        let hsdf = sdf.hsdf();
        let mut milp_formulation = MilpFormulation::new(
            Cow::Borrowed(&hsdf),
            |(a, _)| match a {
                0 => 1,
                1 => 2,
                2 => 2,
                _ => unreachable!(),
            },
            |(a, _)| {
                match a {
                    0 => "a",
                    1 => "b",
                    2 => "c",
                    _ => unreachable!(),
                }
                .to_string()
            },
        ).unwrap();
        milp_formulation.model.set_objective(milp_formulation.throughputs[0], grb::ModelSense::Maximize).unwrap();
        milp_formulation.model.optimize().unwrap();
    }
}
