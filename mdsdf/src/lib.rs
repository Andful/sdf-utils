mod py;
pub mod util;
pub mod vector;

use ndarray::Array2;
use std::borrow::Cow;
use util::{bounded_iterator, repetition_vector, BoundedIterator};
use vector::Vector;

#[derive(Debug, Clone)]
pub struct Channel<const N: usize> {
    pub production_rate: Vector<N, usize>,
    pub consumption_rate: Vector<N, usize>,
    pub source: usize,
    pub target: usize,
    pub initial_tokens: Vector<N, isize>,
}

#[derive(Clone, Copy)]
pub struct ChannelIndex(usize);

#[derive(Debug, Clone, Default)]
pub struct Mdsdf<const N: usize> {
    n_actors: usize,
    channels: Vec<Channel<N>>,
}

impl<const N: usize> Mdsdf<N> {
    pub fn new(n_actors: usize) -> Self {
        Self {
            n_actors,
            channels: Default::default(),
        }
    }

    pub fn add_channel(&mut self, channel: Channel<N>) -> ChannelIndex {
        debug_assert!(channel.source < self.n_actors);
        debug_assert!(channel.target < self.n_actors);
        let c = ChannelIndex(self.channels.len());
        self.channels.push(channel);
        c
    }

    pub fn get_channel(&self, ChannelIndex(i): ChannelIndex) -> &Channel<N> {
        &self.channels[i]
    }

    pub fn hsdf(&self) -> Hsdf<N> {
        let mut rv: Box<[Vector<N, usize>]> =
            vec![Vector::<N, usize>::default(); self.n_actors].into_boxed_slice();
        for d in 0..N {
            let mut topology_matrix = Array2::<i32>::zeros((self.channels.len(), self.n_actors));
            for (
                Channel {
                    production_rate,
                    consumption_rate,
                    source,
                    target,
                    ..
                },
                mut row,
            ) in self.channels.iter().zip(topology_matrix.rows_mut())
            {
                row[*source] += production_rate[d] as i32;
                row[*target] -= consumption_rate[d] as i32;
            }

            for (a, b) in rv
                .iter_mut()
                .zip(repetition_vector(&topology_matrix).expect("sdf is not consistent"))
            {
                a[d] = b;
            }
        }

        Hsdf {
            repetition_vector: rv,
            mdsdf: Cow::Borrowed(self),
        }
    }

    pub fn into_hsdf(self) -> Hsdf<'static, N> {
        let Hsdf {
            repetition_vector, ..
        } = self.hsdf();
        Hsdf {
            repetition_vector,
            mdsdf: Cow::Owned(self),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Hsdf<'a, const N: usize> {
    pub repetition_vector: Box<[Vector<N, usize>]>,
    pub mdsdf: Cow<'a, Mdsdf<N>>,
}

impl<const N: usize> Hsdf<'_, N> {
    pub fn actors(&self) -> HsdfActors<'_, N> {
        HsdfActors {
            hsdf: self,
            actor: 0,
            indicies: bounded_iterator(Cow::Borrowed(&self.repetition_vector[0])),
        }
    }

    pub fn channels(&self) -> HsdfChannels<'_, N, impl Iterator<Item = Channel<N>> + '_> {
        let channels = self.mdsdf.channels.iter().map(Clone::clone);
        HsdfChannels::new(self, channels)
    }
}

pub struct HsdfActors<'a, const N: usize> {
    hsdf: &'a Hsdf<'a, N>,
    actor: usize,
    indicies: BoundedIterator<'a, N>,
}

impl<const N: usize> Iterator for HsdfActors<'_, N> {
    type Item = (usize, Vector<N, usize>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.actor >= self.hsdf.mdsdf.n_actors {
                return None;
            }
            if let Some(index) = self.indicies.next() {
                return Some((self.actor, index));
            } else {
                self.actor += 1;
                if self.actor < self.hsdf.mdsdf.n_actors {
                    self.indicies =
                        bounded_iterator(Cow::Borrowed(&self.hsdf.repetition_vector[self.actor]))
                } else {
                    return None;
                }
            }
        }
    }
}

pub struct HsdfChannel<const N: usize> {
    pub source: (usize, Vector<N, usize>),
    pub target: (usize, Vector<N, usize>),
    pub initial_tokens: Vector<N, isize>,
}

pub struct HsdfChannels<'a, const N: usize, Channels>
where
    Channels: Iterator<Item = Channel<N>>,
{
    pub current_channel: Option<Channel<N>>,
    pub channels: Channels,
    pub hsdf: &'a Hsdf<'a, N>,
    pub bounded_iterator: BoundedIterator<'static, N>,
}

impl<'a, const N: usize, Channels> HsdfChannels<'a, N, Channels>
where
    Channels: Iterator<Item = Channel<N>>,
{
    fn new(hsdf: &'a Hsdf<'a, N>, channels: Channels) -> Self {
        let mut result = Self {
            current_channel: None,
            channels,
            hsdf,
            bounded_iterator: bounded_iterator(Cow::Owned(Default::default())), //TODO
        };
        result.iterate_channel();
        result
    }

    fn iterate_channel(&mut self) {
        self.current_channel = self.channels.next();
        self.bounded_iterator = if let Some(current_channel) = &self.current_channel {
            bounded_iterator(Cow::Owned::<Vector<N, usize>>(
                self.hsdf.repetition_vector[current_channel.source]
                    * current_channel.production_rate,
            ))
        } else {
            return;
        }
    }
}

impl<'a, const N: usize, Channels> Iterator for HsdfChannels<'a, N, Channels>
where
    Channels: Iterator<Item = Channel<N>>,
{
    type Item = HsdfChannel<N>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let current_channel = self.current_channel.as_mut()?;
            let Some(index) = self.bounded_iterator.next() else {
                self.iterate_channel();
                continue;
            };
            let index = index.map(|e| e as isize);
            let tokens = index.clone() + current_channel.initial_tokens;
            let production_rate = current_channel.production_rate.map(|e| e as isize);
            let consumption_rate = current_channel.consumption_rate.map(|e| e as isize);
            let rotation = self.hsdf.repetition_vector[current_channel.target].map(|e| e as isize)
                * consumption_rate.clone();
            let target_index =
                (tokens.clone().modulo(rotation.clone())).flooring_div(consumption_rate);
            let source_index = index.flooring_div(production_rate);
            let initial_tokens = tokens.flooring_div(rotation);

            return Some(HsdfChannel {
                source: (current_channel.source, source_index.map(|e| e as usize)),
                target: (current_channel.target, target_index.map(|e| e as usize)),
                initial_tokens,
            });
        }
    }
}
