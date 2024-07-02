#![feature(iterator_try_collect)]

use mdsdf::{util::bounded_iterator, vector::Vector, Channel, ChannelIndex};
use milp_formulation::{ExecutionTimeT, MilpFormulation, NameT};
use num::integer::gcd;
use std::{borrow::Cow, collections::BTreeMap, isize};

pub struct BufferedMrsdf<
    'a,
    'b: 'a,
    const N: usize,
    ExecutionTime: ExecutionTimeT<N>,
    Name: NameT<N>,
> {
    pub milp: &'a mut MilpFormulation<'b, N, ExecutionTime, Name>,
}

impl<'a, 'b, const N: usize, ExecutionTime: ExecutionTimeT<N>, Name: NameT<N>>
    BufferedMrsdf<'a, 'b, N, ExecutionTime, Name>
{
    pub fn new(milp: &'a mut MilpFormulation<'b, N, ExecutionTime, Name>) -> Self {
        Self { milp }
    }

    pub fn add_buffer(
        &mut self,
        channel: ChannelIndex,
        buffer_size: Vector<N, grb::Expr>,
        name: &str,
    ) -> grb::Result<()> {
        use grb::prelude::*;

        let Channel {
            production_rate: consumption_rate,
            consumption_rate: production_rate,
            source: target,
            target: source,
            ..
        } = self.milp.hsdf.mdsdf.get_channel(channel);

        assert!(
            self.milp.hsdf.repetition_vector[*source] * *production_rate
                == self.milp.hsdf.repetition_vector[*target] * *consumption_rate,
            "Repetition vector would change"
        );
        let model = &mut self.milp.model;
        let (denominator, scale) = production_rate
            .iter()
            .zip(consumption_rate.iter())
            .map(|(a, b)| {
                let scale = gcd(*a, *b);
                ((*a / scale) * (*b / scale), scale)
            })
            .fold(
                (Vec::new(), Vec::new()),
                |(mut denominator, mut scale), (a, b)| {
                    denominator.push(a);
                    scale.push(b);
                    (denominator, scale)
                },
            );
        let denominator: Vector<N, usize> = <[usize; N]>::try_from(denominator).unwrap().into();
        let scale: Vector<N, usize> = <[usize; N]>::try_from(scale).unwrap().into();
        let one_over_denominator = Vector::from([1.0; N]) / denominator.map(|e| e as f64);

        let mut memoized: [BTreeMap<usize, grb::Var>; N] = (0..N)
            .map(|_| Default::default())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        for index in bounded_iterator(Cow::Borrowed(&denominator)) {
            for (d, tokens) in index.iter().enumerate() {
                let den = denominator[d];
                let name = format!("symbolic_token_{name}_{d}=(d+{tokens})/{den}");
                let v = add_intvar!(model, bounds: 0.., name: &name)?;
                memoized[d].insert(*tokens, v.clone());

                let tokens = *tokens as f64;
                model.add_constr(
                    &name,
                    c!(
                        v <= (buffer_size[d].clone() * (1.0 / (scale[d] as f64)) + tokens)
                            * one_over_denominator[d]
                    ),
                )?;
            }
        }

        for si in bounded_iterator(Cow::Borrowed(&self.milp.hsdf.repetition_vector[*source])) {
            for ti in bounded_iterator(Cow::Borrowed(&self.milp.hsdf.repetition_vector[*target])) {
                let tokens = (*production_rate * si).map(|e| e as isize)
                    - (*consumption_rate * ti).map(|e| e as isize)
                    + ((scale * denominator).map(|e| e as isize)
                        - consumption_rate.map(|e| e as isize));
                //tokens is divisible by scale
                debug_assert_eq!(tokens.modulo(scale.map(|e| e as isize)), [0; N].into());
                let to_floor_tokens = (tokens / scale.map(|e| e as isize))
                    .modulo(denominator.map(|e| e as isize))
                    .map(|e| e as usize);
                let hsdf_tokens = tokens.flooring_div((scale * denominator).map(|e| e as isize));
                //let MilpFormulation { u, throughputs, model, execution_time, .. } = &mut self.milp_formulation;
                let u = &self.milp.u;
                let throughputs = &self.milp.throughputs;
                let execution_time = &mut self.milp.execution_time;
                let u_source = u.get(&(*source, si)).unwrap().clone();
                let u_target = u.get(&(*target, ti)).unwrap().clone();
                for d in 0..N {
                    let tokens =
                        memoized[d].get(&to_floor_tokens[d]).unwrap().clone() + hsdf_tokens[d];
                    let throuput = throughputs[d];
                    let et = execution_time.execution_time((*source, si));

                    model.add_constr("", c!(u_target >= u_source + et * throuput - tokens))?;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grb::prelude::*;
    use mdsdf::Mdsdf;
    #[test]
    fn test() {
        let sizes = (7, 3);
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

        let channel1 = sdf.add_channel(Channel {
            production_rate: [2].into(),
            consumption_rate: [3].into(),
            source: 0,
            target: 1,
            initial_tokens: [0].into(),
        });
        let channel2 = sdf.add_channel(Channel {
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
        struct ExecutionTime;
        impl ExecutionTimeT<1> for ExecutionTime {
            fn execution_time(&self, (i, _): (usize, Vector<1, usize>)) -> usize {
                [1, 2, 2][i]
            }
        }
        struct Name;
        impl NameT<1> for Name {
            fn name(&self, (i, j): (usize, Vector<1, usize>)) -> String {
                let n = ["a", "b", "c"][i];
                format!("{}({})", n, j[0])
            }
        }
        let mut milp = MilpFormulation::new(Cow::Borrowed(&hsdf), ExecutionTime, Name).unwrap();
        let mut buffered = BufferedMrsdf::new(&mut milp);

        let model = &mut buffered.milp.model;
        let buffer1 = add_ctsvar!(model, name: "buffer1", bounds: 0..sizes.0).unwrap();
        let buffer2 = add_ctsvar!(model, name: "buffer2", bounds: 0..sizes.1).unwrap();
        buffered
            .add_buffer(channel1, [buffer1.into()].into(), "buffer1")
            .unwrap();
        buffered
            .add_buffer(channel2, [buffer2.into()].into(), "buffer2")
            .unwrap();

        let model = &mut buffered.milp.model;
        let throughput = &buffered.milp.throughputs[0];
        model
            .set_objective(throughput, grb::ModelSense::Maximize)
            .unwrap();
        model.optimize().unwrap();
    }

    /*#[test]
    fn h263() {
        let names: BTreeMap<usize, &str> = ["vld", "iq", "idct", "mc"]
            .iter()
            .map(|e| *e)
            .enumerate()
            .collect();
        let execution_times: BTreeMap<usize, usize> = [39027, 559, 2486, 16437]
            .iter()
            .map(Clone::clone)
            .enumerate()
            .collect();
        let mut sdf = Mdsdf::new(4);
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
            source: 3,
            target: 3,
            initial_tokens: [1].into(),
        });

        sdf.add_channel(Channel {
            production_rate: [594].into(),
            consumption_rate: [1].into(),
            source: 0,
            target: 1,
            initial_tokens: [0].into(),
        });
        sdf.add_channel(Channel {
            production_rate: [1].into(),
            consumption_rate: [1].into(),
            source: 1,
            target: 2,
            initial_tokens: [0].into(),
        });
        sdf.add_channel(Channel {
            production_rate: [1].into(),
            consumption_rate: [594].into(),
            source: 2,
            target: 3,
            initial_tokens: [0].into(),
        });

        let hsdf = sdf.hsdf();
        let execution_time = Box::new(move |(s, _)| *execution_times.get(&s).unwrap());
        let name = Box::new(move |(s, si): (usize, Vector<1, usize>)| {
            format!("{}({})", names.get(&s).unwrap().to_string(), si[0])
        });
        let mut milp = MilpFormulation::new(Cow::Borrowed(&hsdf), execution_time, name).unwrap();
        let mut buffered = BufferedMrsdf::new(&mut milp);

        let model = &mut buffered.milp.model;
        let buffer1 = add_ctsvar!(model, name: "buffer1", bounds: 0..4).unwrap();
        let buffer2 = add_ctsvar!(model, name: "buffer2", bounds: 0..3).unwrap();
        buffered
            .add_buffer(Channel {
                production_rate: [3].into(),
                consumption_rate: [2].into(),
                source: 1,
                target: 0,
                initial_tokens: [buffer1.into()].into(),
            })
            .unwrap();
        buffered
            .add_buffer(Channel {
                production_rate: [2].into(),
                consumption_rate: [1].into(),
                source: 2,
                target: 1,
                initial_tokens: [buffer2.into()].into(),
            })
            .unwrap();

        let model = &mut buffered.milp.model;
        let throughput = &buffered.milp.throughputs[0];
        model
            .set_objective(throughput, grb::ModelSense::Maximize)
            .unwrap();
        model.optimize().unwrap();
    }*/
}
