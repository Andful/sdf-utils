use std::{
    convert::TryInto,
    ops::{Add, Div, Index, IndexMut, Mul, Rem, Sub},
};

use num::Zero;
use pyo3::{types::PyAnyMethods, FromPyObject};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Vector<const N: usize, E>([E; N]);

impl<'a, const N: usize, E: FromPyObject<'a>> FromPyObject<'a> for Vector<N, E> {
    fn extract_bound(ob: &pyo3::Bound<'a, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let e: [E; N] = ob.extract()?;
        Ok(e.into())
    }
}

impl<const N: usize, E, F> From<F> for Vector<N, E>
where
    [E; N]: From<F>,
{
    fn from(value: F) -> Self {
        Self(<[E; N]>::from(value))
    }
}

impl<const N: usize, E> Default for Vector<N, E>
where
    E: Default,
{
    fn default() -> Self {
        let v: [E; N] = match (0..N).map(|_| E::default()).collect::<Vec<_>>().try_into() {
            Ok(v) => v,
            Err(_) => unreachable!(),
        };
        Self(v)
    }
}

impl<const N: usize, E1, E2> Add<Vector<N, E2>> for Vector<N, E1>
where
    E1: Add<E2>,
{
    type Output = Vector<N, E1::Output>;
    fn add(self, rhs: Vector<N, E2>) -> Self::Output {
        let e1 = Vec::from(self.0);
        let e2 = Vec::from(rhs.0);
        match e1
            .into_iter()
            .zip(e2.into_iter())
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>()
            .try_into()
        {
            Ok(e) => Vector::<N, E1::Output>(e),
            Err(_) => unreachable!(),
        }
    }
}

impl<const N: usize, E1, E2> Sub<Vector<N, E2>> for Vector<N, E1>
where
    E1: Sub<E2>,
{
    type Output = Vector<N, E1::Output>;
    fn sub(self, rhs: Vector<N, E2>) -> Self::Output {
        let e1 = Vec::from(self.0);
        let e2 = Vec::from(rhs.0);
        match e1
            .into_iter()
            .zip(e2.into_iter())
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>()
            .try_into()
        {
            Ok(e) => Vector::<N, E1::Output>(e),
            Err(_) => unreachable!(),
        }
    }
}

impl<const N: usize, E1, E2> Mul<Vector<N, E2>> for Vector<N, E1>
where
    E1: Mul<E2>,
{
    type Output = Vector<N, E1::Output>;
    fn mul(self, rhs: Vector<N, E2>) -> Self::Output {
        let e1 = Vec::from(self.0);
        let e2 = Vec::from(rhs.0);
        match e1
            .into_iter()
            .zip(e2.into_iter())
            .map(|(a, b)| a * b)
            .collect::<Vec<_>>()
            .try_into()
        {
            Ok(e) => Vector::<N, E1::Output>(e),
            Err(_) => unreachable!(),
        }
    }
}

impl<const N: usize, E1, E2> Div<Vector<N, E2>> for Vector<N, E1>
where
    E1: Div<E2>,
{
    type Output = Vector<N, E1::Output>;
    fn div(self, rhs: Vector<N, E2>) -> Self::Output {
        let e1 = Vec::from(self.0);
        let e2 = Vec::from(rhs.0);
        match e1
            .into_iter()
            .zip(e2.into_iter())
            .map(|(a, b)| a / b)
            .collect::<Vec<_>>()
            .try_into()
        {
            Ok(e) => Vector::<N, E1::Output>(e),
            Err(_) => unreachable!(),
        }
    }
}

impl<const N: usize, E> Index<usize> for Vector<N, E> {
    type Output = E;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const N: usize, E> IndexMut<usize> for Vector<N, E> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const N: usize, E> Vector<N, E>
where
    E: Ord + Zero + Add<Output = E> + Sub<Output = E> + Rem<Output = E> + Div<Output = E> + Clone,
{
    pub fn modulo(self, rhs: Self) -> Vector<N, E> {
        let e1 = Vec::from(self.0);
        let e2 = Vec::from(rhs.0);
        match e1
            .into_iter()
            .zip(e2.into_iter())
            .map(|(a, b): (E, E)| {
                let mut result = a % b.clone();
                if result < E::zero() {
                    result = result + b;
                }
                result
            })
            .collect::<Vec<_>>()
            .try_into()
        {
            Ok(e) => Vector::<N, E>(e),
            Err(_) => unreachable!(),
        }
    }

    pub fn flooring_div(self, rhs: Self) -> Vector<N, E> {
        let mut result = self.clone();
        result = result.clone() - result.clone().modulo(rhs.clone());
        result / rhs
    }
}
impl<const N: usize, E> Vector<N, E>
where
    E: Clone,
{
    pub fn cast<F>(self) -> Vector<N, F>
    where
        F: From<E>,
    {
        let e: [F; N] = match self
            .0
            .to_vec()
            .into_iter()
            .map(|e| F::from(e))
            .collect::<Vec<_>>()
            .try_into()
        {
            Ok(e) => e,
            Err(_) => unreachable!(),
        };

        Vector(e)
    }

    pub fn map<F>(self, f: impl FnMut(E) -> F) -> Vector<N, F> {
        let e: [F; N] = match self
            .0
            .to_vec()
            .into_iter()
            .map(f)
            .collect::<Vec<_>>()
            .try_into()
        {
            Ok(e) => e,
            Err(_) => unreachable!(),
        };

        Vector(e)
    }
}

impl<const N: usize, E> Vector<N, E> {
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut E> {
        self.0.iter_mut()
    }

    pub fn iter(&self) -> impl Iterator<Item = &E> {
        self.0.iter()
    }
}
