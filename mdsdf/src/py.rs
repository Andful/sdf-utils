use crate::{Channel, Hsdf, HsdfChannel, Mdsdf};
use pyo3::prelude::*;
use std::borrow::Cow;

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
struct PySdf2D {
    sdf: Mdsdf<2>,
    names: Vec<String>,
}

#[pymethods]
impl PySdf2D {
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
        production_rate: (usize, usize),
        consumption_rate: (usize, usize),
        initial_tokens: (isize, isize),
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
            production_rate: production_rate.into(),
            consumption_rate: consumption_rate.into(),
            initial_tokens: initial_tokens.into(),
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
                    let pr = production_rate.iter().map(ToString::to_string).collect::<Vec<_>>().join(",");
                    let cr = consumption_rate.iter().map(ToString::to_string).collect::<Vec<_>>().join(",");
                    let it = initial_tokens.iter().map(ToString::to_string).collect::<Vec<_>>().join(",");
                    format!("  L{source} -> L{target} [taillabel=\"({pr})\" label=\"({it})\" headlabel=\"({cr})\"]\n")
                        .into()
                },
            ))
            .chain(std::iter::once(Cow::Borrowed("}\n")))
            .collect()
    }

    fn hsdf(&self) -> PyHsdf2D {
        PyHsdf2D {
            names: self.names.clone(),
            hsdf: self.sdf.clone().into_hsdf(),
        }
    }
}

#[pyclass(name = "Hsdf2D")]
#[derive(Clone)]
struct PyHsdf2D {
    names: Vec<String>,
    hsdf: Hsdf<'static, 2>,
}

#[pymethods]
impl PyHsdf2D {
    fn actors(&self) -> Vec<(String, (usize, usize))> {
        self.hsdf
            .actors()
            .map(|(i, j)| (self.names[i].clone(), (j[0], j[1])))
            .collect()
    }

    fn channels(
        &self,
    ) -> Vec<(
        (String, (usize, usize)),
        (String, (usize, usize)),
        (isize, isize),
    )> {
        self.hsdf
            .channels()
            .map(
                |HsdfChannel {
                     source: (s, si),
                     target: (t, ti),
                     initial_tokens,
                 }| {
                    (
                        (self.names[s].clone(), (si[0], si[1])),
                        (self.names[t].clone(), (ti[0], ti[1])),
                        (initial_tokens[0], initial_tokens[1]),
                    )
                },
            )
            .collect()
    }

    fn dot(&self) -> String {
        std::iter::once(Cow::Borrowed("digraph {\n"))
            .chain(self.hsdf.actors().map(|(i, j)| {
                let name = &self.names[i];
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
                format!("  L{i}_{indicies} [label=\"{name}({label})\"]\n").into()
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
                    format!("  L{si}_{sj} -> L{ti}_{tj} [label=\"({d})\"]\n").into()
                },
            ))
            .chain(std::iter::once(Cow::Borrowed("}\n")))
            .collect()
    }
}

#[pymodule]
fn mrsdf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySdf>()?;
    m.add_class::<PySdf2D>()?;
    Ok(())
}
