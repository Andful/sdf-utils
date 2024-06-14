use crate::vector::Vector;

use std::borrow::Cow;

use ndarray::prelude::*;
use num::integer::lcm;
use num::Rational32;
use num::Zero;

fn rref<I>(mut out: Array2<Rational32>) -> Array2<Rational32> {
    let mut pivot = 0;
    let (rows, cols) = out.raw_dim().into_pattern();

    'outer: for r in 0..rows {
        if cols <= pivot {
            break;
        }
        let mut i = r;
        while out[[i, pivot]].is_zero() {
            i += 1;
            if i == rows {
                i = r;
                pivot += 1;
                if cols == pivot {
                    break 'outer;
                }
            }
        }

        if r != i {
            let (mut row1, mut row2) = out.multi_slice_mut((s![r, ..], s![i, ..]));
            row1.iter_mut()
                .zip(row2.iter_mut())
                .for_each(|(a, b)| std::mem::swap(a, b));
        }

        let divisor = out[[r, pivot]];
        if !divisor.is_zero() {
            out.row_mut(r).iter_mut().for_each(|e| *e /= divisor);
        }
        for j in 0..rows {
            if j != r {
                let hold = out[[j, pivot]];
                //out.slice_mut(s![r,..]).iter_mut().for_each(|e| *e -= hold * *e);
                for k in 0..cols {
                    let t = out[[r, k]];
                    out[[j, k]] -= hold * t;
                }
            }
        }
        pivot += 1;
    }

    out
}

pub fn repetition_vector(topology_matrix: &Array2<i32>) -> Option<Array1<usize>> {
    let rref = rref::<Rational32>(Array2::<Rational32>::from_shape_fn(
        topology_matrix.raw_dim(),
        |i| Rational32::from(topology_matrix[i]),
    ));
    let (rows, cols) = rref.raw_dim().into_pattern();

    if rref.slice(s![(cols - 1).., ..]) != Array2::<Rational32>::zeros((rows + 1 - cols, cols))
        || rref.slice(s![..(cols - 1), ..(cols - 1)]) != Array2::<Rational32>::eye(cols - 1)
    {
        return None; // rank is not correct
    }
    let unnormalized = rref.slice(s![..(cols - 1), cols - 1]);

    let multiplier = unnormalized
        .iter()
        .map(Rational32::denom)
        .fold(1, |cd, e| lcm(cd, *e));

    Some(Array1::<usize>::from_iter(
        unnormalized
            .iter()
            .map(|e| *(-e * multiplier).numer() as usize)
            .chain(std::iter::once(multiplier as usize)),
    ))
}

pub struct BoundedIterator<'a, const N: usize> {
    index: Vector<N, usize>,
    bound: Cow<'a, Vector<N, usize>>,
}

impl<const N: usize> BoundedIterator<'_, N> {
    fn increment(&mut self) {
        for d in 0..N {
            if self.index[d] + 1 == self.bound[d] {
                self.index[d] = 0;
            } else {
                self.index[d] += 1;
                return;
            }
        }
        self.index[0] = self.bound[0]; // Mark that the iterator is done
    }
}

impl<const N: usize> Iterator for BoundedIterator<'_, N> {
    type Item = Vector<N, usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index[0] == self.bound[0] {
            return None;
        }
        let result = self.index.clone();
        self.increment();
        Some(result)
    }
}

pub fn bounded_iterator<const N: usize>(
    bound: Cow<'_, Vector<N, usize>>,
) -> BoundedIterator<'_, N> {
    BoundedIterator {
        index: Vector::<N, usize>::default(),
        bound,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test1() {
        let topology_matrix = arr2(&[[2, -1, 0], [0, 1, -1], [2, 0, -1], [2, 0, -1]]);
        let repetition_vector = repetition_vector(&topology_matrix);

        assert_eq!(repetition_vector, Some(arr1(&[1, 2, 2])));
    }

    #[test]
    fn test2() {
        let topology_matrix = arr2(&[[2, -3, 0], [1, 0, -1]]);
        let repetition_vector = repetition_vector(&topology_matrix);

        assert_eq!(repetition_vector, Some(arr1(&[3, 2, 3])));
    }
}
