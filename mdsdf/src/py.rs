use crate::{Channel, ChannelIndex, Hsdf, HsdfChannel, Mdsdf};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::result;

#[pyclass(name = "Sdf")]
#[derive(Clone)]
struct PySdf {
    sdf: Mdsdf<1>,
    names: Vec<String>,
}

#[pymethods]
impl PySdf {
    #[new]
    fn new(names: Vec<String>) -> Self {
        Self {
            sdf: Mdsdf::new(names.len()),
            names,
        }
    }

    fn add_channel(
        &mut self,
        source: &str,
        target: &str,
        production_rate: usize,
        consumption_rate: usize,
        initial_tokens: isize,
    ) {
        self.sdf.add_channel(Channel {
            source: self
                .names
                .iter()
                .enumerate()
                .filter(|e| e.1 == source)
                .map(|e| e.0)
                .next()
                .unwrap(),
            target: self
                .names
                .iter()
                .enumerate()
                .filter(|e| e.1 == target)
                .map(|e| e.0)
                .next()
                .unwrap(),
            production_rate: [production_rate].into(),
            consumption_rate: [consumption_rate].into(),
            initial_tokens: [initial_tokens].into(),
        });
    }

    fn dot(&self) -> String {
        std::iter::once(Cow::Borrowed("digraph {\n"))
            .chain(
                self.names
                    .iter()
                    .enumerate()
                    .map(|(i, n)| format!("  L{i} [label=\"{n}\"]\n").into()),
            )
            .chain(self.sdf.channels.iter().map(
                |Channel {
                     production_rate,
                     consumption_rate,
                     source,
                     target,
                     initial_tokens,
                 }| {
                    format!(
                        "  L{source} -> L{target} [taillabel={} label={} headlabel={}]\n",
                        production_rate[0], initial_tokens[0], consumption_rate[0]
                    )
                    .into()
                },
            ))
            .chain(std::iter::once(Cow::Borrowed("}\n")))
            .collect()
    }

    fn hsdf(&self) -> PyHsdf {
        PyHsdf {
            names: self.names.clone(),
            hsdf: self.sdf.clone().into_hsdf(),
        }
    }
}

#[pyclass(name = "Hsdf")]
#[derive(Clone)]
struct PyHsdf {
    names: Vec<String>,
    hsdf: Hsdf<'static, 1>,
}

#[pymethods]
impl PyHsdf {
    fn actors(&self) -> Vec<(String, (usize,))> {
        self.hsdf
            .actors()
            .map(|(i, j)| (self.names[i].clone(), (j[0],)))
            .collect()
    }

    fn channels(&self) -> Vec<((String, (usize,)), (String, (usize,)), (isize,))> {
        self.hsdf
            .channels()
            .map(
                |HsdfChannel {
                     source: (s, si),
                     target: (t, ti),
                     initial_tokens: d,
                 }| {
                    (
                        (self.names[s].clone(), (si[0],)),
                        (self.names[t].clone(), (ti[0],)),
                        (d[0],),
                    )
                },
            )
            .collect()
    }

    fn dot(&self) -> String {
        std::iter::once(Cow::Borrowed("digraph {\n"))
            .chain(self.hsdf.actors().map(|(i, j)| {
                let name = &self.names[i];
                format!("  L{i}_{0} [label=\"{name}({0})\"]\n", j[0]).into()
            }))
            .chain(self.hsdf.channels().map(
                |HsdfChannel {
                     source: (si, sj),
                     target: (ti, tj),
                     initial_tokens: d,
                 }| {
                    format!("  L{si}_{} -> L{ti}_{} [label={}]\n", sj[0], tj[0], d[0]).into()
                },
            ))
            .chain(std::iter::once(Cow::Borrowed("}\n")))
            .collect()
    }
}

#[pyclass(name = "Sdf2D")]
#[derive(Clone)]
pub struct PySdf2D {
    pub sdf: Mdsdf<2>,
    pub id_map: Py<PyDict>,
}

#[pyclass(name = "Channel")]
#[derive(Clone)]
struct PyChannelIndex(ChannelIndex);

#[pymethods]
impl PySdf2D {
    #[new]
    fn new(ids: Py<PySet>) -> PyResult<Self> {
        let (l, id_map) = Python::with_gil(|py| {
            let l = ids.bind(py).len();
            let id_map = PyDict::new_bound(py);
            for (i, id) in ids.bind(py).iter().enumerate() {
                id_map.set_item(id, i)?;
            }
            PyResult::Ok((l, id_map.as_unbound().clone()))
        })?;
        Ok(Self {
            sdf: Mdsdf::new(l),
            id_map,
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
        source: Py<PyAny>,
        target: Py<PyAny>,
        production_rate: (usize, usize),
        consumption_rate: (usize, usize),
        initial_tokens: (isize, isize),
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
            production_rate: production_rate.into(),
            consumption_rate: consumption_rate.into(),
            initial_tokens: initial_tokens.into(),
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
        let index = self
            .sdf
            .channels
            .iter()
            .enumerate()
            .try_fold(None, |init, result| {
                let Channel { source, target, .. } = &result.1;
                if id_map
                    .get_item(source)?
                    .expect("unreachable")
                    .eq(&channel.0)?
                    && id_map
                        .get_item(target)?
                        .expect("unreachable")
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
            initial_tokens: [0, 0].into(),
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

    fn induce<'py>(&self, py: Python<'py>, sub_ids: Py<PySet>) -> PyResult<PySdf2D> {
        let mut result = PySdf2D::new(sub_ids)?;
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

    fn dot<'py>(&self, py: Python<'py>) -> PyResult<String> {
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

    fn hsdf<'py>(&self, py: Python<'py>) -> PyResult<PyHsdf2D> {
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

        Ok(PyHsdf2D {
            id_map: unsorted_id_map.into_iter().map(|(_, e)| e).collect(),
            hsdf: self.sdf.clone().into_hsdf(),
        })
    }
}

#[pyclass(name = "Hsdf2D")]
#[derive(Clone)]
struct PyHsdf2D {
    id_map: Vec<Py<PyAny>>,
    hsdf: Hsdf<'static, 2>,
}

#[pymethods]
impl PyHsdf2D {
    fn actors<'py>(&self, py: Python<'py>) -> PyResult<Vec<(Bound<'py, PyAny>, (usize, usize))>> {
        self.hsdf
            .actors()
            .map(|(i, j)| Ok((self.id_map[i].bind(py).clone(), (j[0], j[1]))))
            .try_collect()
    }

    fn channels<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<
        Vec<(
            (Bound<'py, PyAny>, (usize, usize)),
            (Bound<'py, PyAny>, (usize, usize)),
            (isize, isize),
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
                        (self.id_map[s].bind(py).clone(), (si[0], si[1])),
                        (self.id_map[t].bind(py).clone(), (ti[0], ti[1])),
                        (initial_tokens[0], initial_tokens[1]),
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

#[pymodule]
fn mdsdf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySdf>()?;
    m.add_class::<PySdf2D>()?;
    Ok(())
}
