use alloc::vec::Vec;
use core::cmp::Ordering::Less;
use core::cmp::Ordering;

fn partition<F>(
    a: &mut [usize],
    left: usize,
    right: usize,
    pivot: usize,
    compare: &mut F,
)
-> usize
where
    F: FnMut(&usize, &usize) -> Ordering {
    a.swap(pivot, right);
    let mut store_index = left;
    for i in left..right {
        if compare(&a[i], &a[right]) == Less {
            a.swap(store_index, i);
            store_index += 1;
        }
    }
    a.swap(right, store_index);
    store_index
}

fn pivot_index(left: usize, right: usize) -> usize {
    return left + (right - left) / 2;
}

fn select<F>(a: &mut [usize], mut left: usize, mut right: usize, n: usize, compare: &mut F)
where
    F: FnMut(&usize, &usize) -> Ordering {
    loop {
        if left == right {
            break;
        }
        let mut pivot = pivot_index(left, right);
        pivot = partition::<F>(a, left, right, pivot, compare);
        if n == pivot {
            break;
        } else if n < pivot {
            right = pivot - 1;
        } else {
            left = pivot + 1;
        }
    }
}

// Rearranges the elements of 'a' such that the element at index 'n' is
// the same as it would be if the array were sorted, smaller elements are
// to the left of it and larger elements are to its right.
pub fn nth_element<F>(a: &mut [usize], n: usize, compare: &mut F)
where
    F: FnMut(&usize, &usize) -> Ordering
{
    select::<F>(a, 0, a.len() - 1, n, compare);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_nth_element() {
        let size = 100;
        let mut rng = rand::thread_rng();

        for _ in 0..2000 {
            let n: usize = rng.gen_range(0..size);
            let vector = (0..size).map(|_| rng.gen()).collect::<Vec<f64>>();
            let mut indices = (0..size).collect::<Vec<usize>>();
            let mut compare = |i1: &usize, i2: &usize| {
                vector[*i1].partial_cmp(&vector[*i2]).unwrap()
            };

            nth_element(&mut indices, n, &mut compare);

            for i in 0..n {
                assert!(vector[indices[i]] <= vector[n]);
            }
            for i in (n + 1)..indices.len() {
                assert!(vector[indices[i]] >= vector[n]);
            }
        }
    }
}
