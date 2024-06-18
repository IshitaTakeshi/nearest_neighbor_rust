#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
use log::info; // Use log crate when building application

#[cfg(feature = "std")]
use std::{println as info};

#[macro_use]
extern crate alloc;

mod log2;

use crate::log2::log2;
use num_traits::float::FloatCore;
use core::cmp::Ordering;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use crate::alloc::string::ToString;

use nalgebra::base::dimension::Const;
use nalgebra::{ArrayStorage, U1};

pub type Vector<const D: usize> = nalgebra::Matrix<f64, Const<D>, U1, ArrayStorage<f64, D, 1>>;

fn divide<const D: usize>(
    indices: &mut [usize],
    data: &[Vector<D>],
    dim: usize,
) -> (f64, Vec<usize>, Vec<usize>) {
    let k = if indices.len() % 2 == 0 {
        indices.len() / 2
    } else {
        indices.len() / 2 + 1
    };

    let cmp = |i1: &usize, i2: &usize| -> Ordering {
        let d1: &Vector<D> = &data[*i1];
        let d2: &Vector<D> = &data[*i2];
        d1[dim].partial_cmp(&d2[dim]).unwrap()
    };

    let (indices_l, index, indices_r) = indices.select_nth_unstable_by(k, cmp);
    let indices_l = indices_l.iter().map(|&s| s).collect::<Vec<usize>>();
    let mut indices_r = indices_r.iter().map(|&s| s).collect::<Vec<usize>>();
    indices_r.push(*index);
    (data[*index][dim], indices_l, indices_r)
}

fn node_index_to_depth(node_index: usize) -> usize {
    log2(node_index as u64) as usize
}

fn node_index_to_dim<const D: usize>(node_index: usize) -> usize {
    node_index_to_depth(node_index) % D
}

fn calc_depth(data_size: usize, leaf_size: usize) -> u64 {
    assert!(data_size > 0);
    log2((data_size - 1) as u64 / leaf_size as u64) + 1
}

fn construct<const D: usize>(
    data: &[Vector<D>],
    leaf_size: usize,
) -> (Vec<Option<f64>>, BTreeMap<usize, Vec<usize>>) {
    assert!(data.len() >= leaf_size);
    let indices = (0..data.len()).collect::<Vec<usize>>();
    let depth = calc_depth(data.len(), leaf_size);
    let n_leaves = 2usize.pow(depth as u32);
    let mut boundaries = (0..n_leaves).map(|_| None).collect::<Vec<Option<f64>>>();
    let mut leaf_nodes = BTreeMap::<usize, Vec<usize>>::new();

    let mut stack = Vec::from([(indices, 1)]);
    while stack.len() != 0 {
        let (mut indices, node_index) = stack.pop().unwrap();

        if indices.len() <= leaf_size {
            info!("Insert: node = {:2},  indices = {:?}", node_index, indices);
            leaf_nodes.insert(node_index, indices);
            continue;
        }

        let dim = node_index_to_dim::<D>(node_index);

        let (boundary, indices_l, indices_r) = divide(&mut indices, data, dim);

        boundaries[node_index] = Some(boundary);
        stack.push((indices_l, node_index * 2 + 0));
        stack.push((indices_r, node_index * 2 + 1));
    }
    (boundaries, leaf_nodes)
}

fn squared_euclidean<const D: usize>(a: &Vector<D>, b: &Vector<D>) -> f64 {
    let d = a - b;
    d.dot(&d)
}

fn find_nearest<const D: usize>(
    query: &Vector<D>,
    indices: &[usize],
    data: &[Vector<D>],
) -> (Option<usize>, f64) {
    let mut min_distance = f64::INFINITY;
    let mut argmin = None;
    for &index in indices {
        let d = squared_euclidean(query, &data[index]);
        if d < min_distance {
            min_distance = d;
            argmin = Some(index);
        }
    }
    (argmin, min_distance)
}

fn find_leaf<const D: usize>(query: &Vector<D>, boundaries: &[Option<f64>]) -> usize {
    let mut node_index = 1;
    let mut dim: usize = 0;
    while node_index < boundaries.len() {
        let Some(boundary) = boundaries[node_index] else {
            break;
        };

        node_index = if query[(dim, 0)] < boundary {
            node_index * 2 + 0
        } else {
            node_index * 2 + 1
        };
        dim = (dim + 1) % D;
    }
    node_index
}

fn print_tree<const D: usize>(
    boundaries: &[Option<f64>],
    leaf_nodes: &BTreeMap<usize, Vec<usize>>,
    data: &[Vector<D>]
) {
    let mut stack = Vec::from([1]);
    while stack.len() != 0 {
        let node_index = stack.pop().unwrap();
        let depth = node_index_to_depth(node_index);

        if let Some(indices) = leaf_nodes.get(&node_index) {
            let samples = indices.iter().map(|&i| data[i]).collect::<Vec<Vector<D>>>();
            let s = " ".repeat(2 * depth + 5);
            info!("{} {:3} {:?}", s, node_index, samples);
            continue;
        };

        let b = match boundaries[node_index] {
            None => "".to_string(),
            Some(boundary) => format!("{:.5}", boundary),
        };
        let dim = node_index_to_dim::<D>(node_index);
        info!("{} {:3} : {} : {}", " ".repeat(2 * depth), node_index, dim, b);

        stack.push(node_index * 2 + 0);
        stack.push(node_index * 2 + 1);
    }
}

fn search<const D: usize>(
    query: &Vector<D>,
    boundaries: &[Option<f64>],
    leaf_nodes: &BTreeMap<usize, Vec<usize>>,
    data: &[Vector<D>],
) -> (usize, f64) {
    let leaf_index = find_leaf(query, boundaries);
    let Some(indices) = leaf_nodes.get(&leaf_index) else {
        panic!("Leaf corresponding to leaf_index {} not found", leaf_index);
    };

    let (maybe_argmin, distance) = find_nearest(query, &indices, data);
    (maybe_argmin.unwrap(), distance)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn search_and_print<const D: usize>(
        query: &Vector<D>,
        boundaries: &[Option<f64>],
        leaf_nodes: &BTreeMap<usize, Vec<usize>>,
        data: &[Vector<D>],
    ) {
        let (argmin, distance) = search(query, boundaries, leaf_nodes, data);
        info!(
            "query = {:?}, found = {:?}, distance = {}",
            query, data[argmin], distance
        );
    }

    #[test]
    fn run() {
        let data = [
            [2., 2.],
            [3., 7.],
            [3., 13.],
            [3., 18.],
            [5., 10.],
            [6., 15.],
            [7., 6.],
            [8., 3.],
            [8., 18.],
            [10., 8.],
            [10., 11.],
            [10., 14.],
            [11., 4.],
            [11., 6.],
            [13., 1.],
            [13., 10.],
            [13., 16.],
            [14., 7.],
            [14., 19.],
            [15., 4.],
            [15., 12.],
            [17., 17.],
            [18., 5.],
            [18., 8.],
            [18., 10.],
        ];

        let data = &data
            .iter()
            .map(|s| (*s).into())
            .collect::<Vec<nalgebra::SVector<f64, 2>>>();
        let leaf_size = 1;

        let (boundaries, leaf_nodes) = construct(data, leaf_size);
        print_tree(&boundaries, &leaf_nodes, data);
        for query in data.iter() {
            search_and_print(query, &boundaries, &leaf_nodes, data);
        }

        // search_and_print(&Vector::<2>::new(14., 13.), &boundaries, &leaf_nodes, data);
        // search_and_print(&Vector::<2>::new(10., 11.), &boundaries, &leaf_nodes, data);
        // search_and_print(&Vector::<2>::new(2., 11.), &boundaries, &leaf_nodes, data);
    }
}
