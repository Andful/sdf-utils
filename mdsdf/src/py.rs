use crate::{Channel, ChannelIndex, Hsdf, HsdfChannel, Mdsdf};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple};
use std::collections::BTreeSet;

#[pyclass(name = "Channel")]
#[derive(Clone)]
struct PyChannelIndex(ChannelIndex);

#[derive(Clone)]
pub struct PySdf<const N: usize> {
    pub sdf: Mdsdf<N>,
    pub id_map: Py<PyDict>,
}

impl<const N: usize> PySdf<N> {
    fn new<'py>(py: Python<'py>, ids: Bound<'py, PySet>) -> PyResult<Self> {
        let l = ids.len();
        let id_map = PyDict::new_bound(py);
        for (i, id) in ids.iter().enumerate() {
            id_map.set_item(id, i)?;
        }
        Ok(Self {
            sdf: Mdsdf::new(l),
            id_map: id_map.as_unbound().clone(),
        })
    }

    fn __str__<'py>(&self, py: Python<'py>) -> PyResult<String> {
        let s = self
            .id_map
            .bind(py)
            .keys()
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let s = format!("actors = {{{s}}}");

        Ok(s)
    }

    fn add_channel<'py>(
        &mut self,
        py: Python<'py>,
        source: Bound<'py, PyAny>,
        target: Bound<'py, PyAny>,
        production_rate: Bound<'py, PyTuple>,
        consumption_rate: Bound<'py, PyTuple>,
        initial_tokens: Bound<'py, PyTuple>,
    ) -> PyResult<()> {
        self.sdf.add_channel(Channel {
            source: self
                .id_map
                .bind(py)
                .get_item(source)?
                .ok_or(PyValueError::new_err("No such actor"))?
                .extract()?,
            target: self
                .id_map
                .bind(py)
                .get_item(target)?
                .ok_or(PyValueError::new_err("No such actor"))?
                .extract()?,
            production_rate: production_rate.extract::<[usize; N]>()?.into(),
            consumption_rate: consumption_rate.extract::<[usize; N]>()?.into(),
            initial_tokens: initial_tokens.extract::<[isize; N]>()?.into(),
        });
        Ok(())
    }

    fn add_actor<'py>(&mut self, py: Python<'py>, e: Bound<'py, PyAny>) -> PyResult<()> {
        self.id_map.bind(py).set_item(e, self.sdf.n_actors)?;
        self.sdf.n_actors += 1;
        Ok(())
    }

    fn sources<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let targets: BTreeSet<usize> = self
            .sdf
            .channels
            .iter()
            .map(|Channel { target, .. }| *target)
            .collect();
        self.id_map
            .bind(py)
            .iter()
            .filter_map(|(e, i)| {
                let i: usize = match i.extract() {
                    Ok(i) => i,
                    Err(e) => return Some(Err(e)),
                };

                if targets.contains(&i) {
                    None
                } else {
                    Some(Ok(e))
                }
            })
            .try_collect::<Vec<_>>()
    }

    fn sinks<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let sources: BTreeSet<usize> = self
            .sdf
            .channels
            .iter()
            .map(|Channel { source, .. }| *source)
            .collect();
        self.id_map
            .bind(py)
            .iter()
            .filter_map(|(e, i)| {
                let i: usize = match i.extract() {
                    Ok(i) => i,
                    Err(e) => return Some(Err(e)),
                };

                if sources.contains(&i) {
                    None
                } else {
                    Some(Ok(e))
                }
            })
            .try_collect::<Vec<_>>()
    }

    fn actors<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        self.id_map.bind(py).keys()
    }

    fn channels<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
        let id_to_hashable = self
            .id_map
            .bind(py)
            .iter()
            .map(|(a, b)| (b.to_owned(), a.to_owned()))
            .try_fold(PyDict::new_bound(py), |d, (a, b)| {
                d.set_item(a, b)?;
                PyResult::Ok(d)
            })?;
        self.sdf
            .channels
            .iter()
            .map(|Channel { source, target, .. }| {
                match (
                    id_to_hashable.get_item(source),
                    id_to_hashable.get_item(target),
                ) {
                    (Ok(e1), Ok(e2)) => Ok((e1.unwrap(), e2.unwrap())),
                    (Err(e), _) => Err(e),
                    (_, Err(e)) => Err(e),
                }
            })
            .try_collect()
    }

    /**
     * Cut channel to be separated by an actor. Initial tokens go ot the target actor side
     */
    fn cut_channel<'py>(
        &mut self,
        py: Python<'py>,
        channel: (Bound<'py, PyAny>, Bound<'py, PyAny>),
        new_actor: Bound<'py, PyAny>,
    ) -> PyResult<()> {
        let id_map = self.id_map.bind(py);
        if let Some(_) = id_map.get_item(&new_actor)? {
            return Err(PyErr::new::<PyValueError, _>("Actor already present"));
        }
        let new_actor_id = id_map.len();
        id_map.set_item(new_actor, new_actor_id)?;
        self.sdf.n_actors = id_map.len();

        let mut id_map = id_map.iter().map(|(a, b)| PyResult::Ok((b.extract::<usize>()?, a))).try_collect::<Vec<(usize, Bound<'py, PyAny>)>>()?;
        id_map.sort_by_key(|(i, _)| *i);
        debug_assert!(id_map.iter().enumerate().all(|(i, (j, _))| i == *j));
        let id_map = id_map.into_iter().map(|(_, a)| a).collect::<Vec<_>>();

        let index = self
            .sdf
            .channels
            .iter()
            .enumerate()
            .try_fold(None, |init, result| {
                let Channel { source, target, .. } = &result.1;
                if id_map[*source]
                    .eq(&channel.0)?
                    && id_map[*target]
                        .eq(&channel.1)?
                {
                    let None = init else {
                        return Err(PyErr::new::<PyValueError, _>("Sdf is a multigraph"));
                    };
                    Ok(Some(result.0))
                } else {
                    Ok(init)
                }
            })?
            .ok_or(PyErr::new::<PyValueError, _>("No such channel"))?;
        let Channel {
            production_rate,
            consumption_rate,
            source,
            target,
            initial_tokens,
        } = self.sdf.channels.remove(index);
        self.sdf.add_channel(Channel {
            production_rate,
            consumption_rate: production_rate,
            source,
            target: new_actor_id,
            initial_tokens: Default::default(),
        });
        self.sdf.add_channel(Channel {
            production_rate,
            consumption_rate,
            source: new_actor_id,
            target,
            initial_tokens,
        });
        Ok(())
    }

    fn induce<'py>(&self, py: Python<'py>, sub_ids: Bound<'py, PySet>) -> PyResult<Self> {
        let mut result = Self::new(py, sub_ids)?;
        let id_to_hashable = self
            .id_map
            .bind(py)
            .iter()
            .map(|(a, b)| (b.to_owned(), a.to_owned()))
            .try_fold(PyDict::new_bound(py), |d, (a, b)| {
                d.set_item(a, b)?;
                PyResult::Ok(d)
            })?;

        for Channel {
            production_rate,
            consumption_rate,
            source,
            target,
            initial_tokens,
        } in self.sdf.channels.iter().map(Clone::clone)
        {
            let (Some(source), Some(target)) = (
                result
                    .id_map
                    .bind(py)
                    .get_item(id_to_hashable.get_item(source).unwrap())?
                    .map(|e| e.extract()),
                result
                    .id_map
                    .bind(py)
                    .get_item(id_to_hashable.get_item(target).unwrap())?
                    .map(|e| e.extract()),
            ) else {
                continue;
            };
            result.sdf.add_channel(Channel {
                production_rate,
                consumption_rate,
                source: source?,
                target: target?,
                initial_tokens,
            });
        }

        Ok(result)
    }

    pub fn dot<'py>(&self, py: Python<'py>) -> PyResult<String> {
        std::iter::once(Ok("digraph {\n".to_string()))
            .chain(
                self.id_map.bind(py)
                    .keys()
                    .iter()
                    .enumerate()
                    .map(|(i, n)| {
                        Ok(format!("  L{i} [label=\"{n}\"]\n").into())
                    }),
            )
            .chain(self.sdf.channels.iter().map(
                |Channel {
                     production_rate,
                     consumption_rate,
                     source,
                     target,
                     initial_tokens,
                 }| {
                    let pr = production_rate.iter().map(ToString::to_string).collect::<Vec<_>>().join(",");
                    let cr = consumption_rate.iter().map(ToString::to_string).collect::<Vec<_>>().join(",");
                    let it = initial_tokens.iter().map(ToString::to_string).collect::<Vec<_>>().join(",");
                    Ok(format!("  L{source} -> L{target} [taillabel=\"({pr})\" label=\"({it})\" headlabel=\"({cr})\"]\n")
                        .into())
                },
            ))
            .chain(std::iter::once(Ok("}\n".to_string())))
            .try_collect()
    }

    fn hsdf<'py>(&self, py: Python<'py>) -> PyResult<PyHsdf<N>> {
        let mut unsorted_id_map = self
            .id_map
            .bind(py)
            .iter()
            .map(|(a, b)| PyResult::Ok((b.extract::<usize>()?, a.as_unbound().clone())))
            .try_collect::<Vec<_>>()?;
        unsorted_id_map.sort_by_key(|(i, _)| *i);
        debug_assert!(unsorted_id_map
            .iter()
            .enumerate()
            .all(|(i, (j, _))| i == *j));

        Ok(PyHsdf {
            id_map: unsorted_id_map.into_iter().map(|(_, e)| e).collect(),
            hsdf: self.sdf.clone().into_hsdf(),
        })
    }
}

#[derive(Clone)]
struct PyHsdf<const N: usize> {
    id_map: Vec<Py<PyAny>>,
    hsdf: Hsdf<'static, N>,
}

impl<const N: usize> PyHsdf<N> {
    fn actors<'py>(&self, py: Python<'py>) -> PyResult<Vec<(Bound<'py, PyAny>, Py<PyTuple>)>> {
        self.hsdf
            .actors()
            .map(|(i, j)| Ok((self.id_map[i].bind(py).clone(), j.into_py(py))))
            .try_collect()
    }

    fn channels<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<
        Vec<(
            (Bound<'py, PyAny>, Py<PyTuple>),
            (Bound<'py, PyAny>, Py<PyTuple>),
            Py<PyTuple>,
        )>,
    > {
        self.hsdf
            .channels()
            .map(
                |HsdfChannel {
                     source: (s, si),
                     target: (t, ti),
                     initial_tokens,
                 }| {
                    Ok((
                        (self.id_map[s].bind(py).clone(), si.into_py(py)),
                        (self.id_map[t].bind(py).clone(), ti.into_py(py)),
                        initial_tokens.into_py(py),
                    ))
                },
            )
            .try_collect()
    }

    fn dot<'py>(&self, py: Python<'py>) -> PyResult<String> {
        std::iter::once(PyResult::Ok("digraph {\n".to_string()))
            .chain(self.hsdf.actors().map(|(i, j)| {
                let name = &self.id_map[i].bind(py).clone();
                let indicies = j
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("_");
                let label = j
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                Ok(format!("  L{i}_{indicies} [label=\"{name}({label})\"]\n"))
            }))
            .chain(self.hsdf.channels().map(
                |HsdfChannel {
                     source: (si, sj),
                     target: (ti, tj),
                     initial_tokens: d,
                 }| {
                    let sj = sj
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("_");
                    let tj = tj
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("_");
                    let d = d
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(",");
                    Ok(format!("  L{si}_{sj} -> L{ti}_{tj} [label=\"({d})\"]\n"))
                },
            ))
            .chain(std::iter::once(Ok("}\n".to_string())))
            .collect()
    }
}

macro_rules! impl_sdf {
    ($sdf_name:ident, $hsdf_name:ident, $n:literal) => {
        #[derive(Clone)]
        #[pyclass]
        pub struct $hsdf_name(pub PyHsdf<$n>);

        #[derive(Clone)]
        #[pyclass]
        pub struct $sdf_name(pub PySdf<$n>);

        #[pymethods]
        impl $sdf_name {
            #[new]
            pub fn new<'py>(py: Python<'py>, ids: Bound<'py, PySet>) -> PyResult<Self> {
                Ok(Self(PySdf::new(py, ids)?))
            }

            pub fn __str__<'py>(&self, py: Python<'py>) -> PyResult<String> {
                self.0.__str__(py)
            }

            pub fn add_channel<'py>(
                &mut self,
                py: Python<'py>,
                source: Bound<'py, PyAny>,
                target: Bound<'py, PyAny>,
                production_rate: Bound<'py, PyTuple>,
                consumption_rate: Bound<'py, PyTuple>,
                initial_tokens: Bound<'py, PyTuple>,
            ) -> PyResult<()> {
                self.0.add_channel(
                    py,
                    source,
                    target,
                    production_rate,
                    consumption_rate,
                    initial_tokens,
                )
            }

            pub fn add_actor<'py>(&mut self, py: Python<'py>, e: Bound<'py, PyAny>) -> PyResult<()> {
                self.0.add_actor(py, e)
            }

            pub fn sources<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
                self.0.sources(py)
            }

            pub fn sinks<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
                self.0.sinks(py)
            }

            pub fn actors<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
                self.0.actors(py)
            }

            pub fn channels<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
                self.0.channels(py)
            }

            pub fn cut_channel<'py>(
                &mut self,
                py: Python<'py>,
                channel: (Bound<'py, PyAny>, Bound<'py, PyAny>),
                new_actor: Bound<'py, PyAny>,
            ) -> PyResult<()> {
                self.0.cut_channel(py, channel, new_actor)
            }

            pub fn induce<'py>(&self, py: Python<'py>, sub_ids: Bound<'py, PySet>) -> PyResult<Self> {
                Ok(Self(self.0.induce(py, sub_ids)?))
            }

            pub fn dot<'py>(&self, py: Python<'py>) -> PyResult<String> {
                self.0.dot(py)
            }

            pub fn hsdf<'py>(&self, py: Python<'py>) -> PyResult<$hsdf_name> {
                Ok($hsdf_name(self.0.hsdf(py)?))
            }
        }

        #[pymethods]
        impl $hsdf_name {
            pub fn actors<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Vec<(Bound<'py, PyAny>, Py<PyTuple>)>> {
                self.0.actors(py)
            }

            pub fn channels<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<
                Vec<(
                    (Bound<'py, PyAny>, Py<PyTuple>),
                    (Bound<'py, PyAny>, Py<PyTuple>),
                    Py<PyTuple>,
                )>,
            > {
                self.0.channels(py)
            }

            pub fn dot<'py>(&self, py: Python<'py>) -> PyResult<String> {
                self.0.dot(py)
            }
        }
    };
}

impl_sdf!(Sdf1D, Hsdf1D, 1);
impl_sdf!(Sdf2D, Hsdf2D, 2);
impl_sdf!(Sdf3D, Hsdf3D, 3);

#[pymodule]
fn mdsdf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Sdf1D>()?;
    m.add_class::<Sdf2D>()?;
    m.add_class::<Sdf3D>()?;
    Ok(())
}
