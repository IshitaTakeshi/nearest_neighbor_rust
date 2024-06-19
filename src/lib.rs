#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
use log::info; // Use log crate when building application

#[cfg(feature = "std")]
use std::{println as info};

#[macro_use]
extern crate alloc;

use num_traits::float::FloatCore;
use core::cmp::Ordering;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use crate::alloc::string::ToString;

use nalgebra::base::dimension::Const;
use nalgebra::{ArrayStorage, U1};

pub type Vector<const D: usize> = nalgebra::Matrix<f64, Const<D>, U1, ArrayStorage<f64, D, 1>>;

fn divide<const D: usize>(
    indices: &mut Vec<usize>,
    data: &[Vector<D>],
    dim: usize,
) -> (f64, Vec<usize>, Vec<usize>) {
    let cmp = |i1: &usize, i2: &usize| -> Ordering {
        let d1: &Vector<D> = &data[*i1];
        let d2: &Vector<D> = &data[*i2];
        d1[dim].partial_cmp(&d2[dim]).unwrap()
    };

    indices.sort_by(cmp);

    let mut k = indices.len() / 2;
    while k > 0 && data[indices[k]][dim] == data[indices[k-1]][dim] {
        k = k - 1;
    }
    let indices_r = indices.split_off(k);
    let indices_l = indices.clone();
    (data[indices_r[0]][dim], indices_l, indices_r)
}

pub struct KdTree<'a, const D: usize> {
    data: &'a [Vector<D>],
    leaf_size: usize,
    boundaries: BTreeMap::<usize, f64>,
    leaves: BTreeMap::<usize, Vec<usize>>,
}

fn panic_leaf_node_not_found<const D: usize>(query: &Vector<D>, leaf_index: usize) -> ! {
    panic!(
        "Leaf node corresponding to query = {:?} node_index = {} not found. \
        There's something wrong in the tree construction. \
        Report the bug to the repository owner.", query, leaf_index)
}

fn find_dim<const D: usize>(node_index: usize) -> usize {
    assert!(node_index > 0);
    let mut i = 2;
    let mut dim = 0;
    while i <= node_index {
        i *= 2;
        dim = (dim + 1) % D;
    }
    dim
}

fn children_near_far(query_element: f64, boundary: f64, node_index: usize) -> (usize, usize) {
    if query_element < boundary {
        (node_index * 2 + 0, node_index * 2 + 1)
    } else {
        (node_index * 2 + 1, node_index * 2 + 0)
    }
}

fn find_within_distance<const D: usize>(
    node_index: usize,
    query: &Vector<D>,
    argmin: &Option<usize>,
    distance: f64,
    data: &[Vector<D>],
    boundaries: &BTreeMap::<usize, f64>,
    leaves: &BTreeMap::<usize, Vec<usize>>
) -> (Option<usize>, f64) {
    println!("find_within_distance with node_index = {}", node_index);
    let mut argmin = *argmin;
    let mut distance = distance;
    let mut stack = Vec::from([(node_index, find_dim::<D>(node_index))]);
    while stack.len() != 0 {
        let (node_index, dim) = stack.pop().unwrap();
        println!("node_index = {}  dim = {}", node_index, dim);
        let Some(&boundary) = boundaries.get(&node_index) else {
            println!("  reached the leaf {}", node_index);
            // If `node_index` is not in boundaries, `node_index` must be a leaf.
            let indices = leaves.get(&node_index).unwrap();
            // Update if the nearest element in the leaf is closer than the current
            // nearest
            let (argmin_candidate, distance_candidate) = find_nearest(query, &indices, data);
            if distance_candidate < distance {
                argmin = argmin_candidate;
                distance = distance_candidate;
            }
            continue;
        };

        let (near, far) = children_near_far(query[(dim, 0)], boundary, node_index);

        let next_dim = (dim + 1) % D;
        stack.push((near, next_dim));

        // Boundary is farther than the nearest element
        if squared_diff(query[(dim, 0)], boundary) > distance {
            println!("squared_diff > distance");
            continue;
        }
        stack.push((far, next_dim));
    }
    (argmin, distance)
}

fn the_other_side_index(node_index: usize) -> usize {
    if node_index % 2 == 0 {
        node_index + 1
    } else {
        node_index - 1
    }
}

fn find_nearest_in_other_areas<const D: usize>(
    query: &Vector<D>,
    argmin: &Option<usize>,
    distance: f64,
    node_index: usize,
    data: &[Vector<D>],
    boundaries: &BTreeMap::<usize, f64>,
    leaves: &BTreeMap::<usize, Vec<usize>>
) -> (Option<usize>, f64) {
    let mut node_index = node_index;
    let mut argmin = *argmin;
    let mut distance = distance;
    while node_index > 1 {
        let the_other_side_index = the_other_side_index(node_index);
        println!("the_other_side_index = {}", the_other_side_index);
        (argmin, distance) = find_within_distance(the_other_side_index, query, &argmin, distance, data, boundaries, leaves);
        node_index = node_index / 2;
    }
    (argmin, distance)
}

impl<'a, const D: usize> KdTree<'a, D> {
    pub fn new(
        data: &'a [Vector<D>],
        leaf_size: usize,
    ) -> Self {
        assert!(data.len() >= leaf_size);
        let indices = (0..data.len()).collect::<Vec<usize>>();
        let mut boundaries = BTreeMap::<usize, f64>::new();
        let mut leaves = BTreeMap::<usize, Vec<usize>>::new();

        let mut stack = Vec::from([(indices, 1, 0)]);
        while stack.len() != 0 {
            let (mut indices, node_index, dim) = stack.pop().unwrap();

            if indices.len() <= leaf_size {
                println!("node_index = {}  leaf elements = {:?}",
                    node_index, indices.iter().map(|&i| data[i]).collect::<Vec<Vector<D>>>());
                leaves.insert(node_index, indices);
                continue;
            }

            let (boundary, indices_l, indices_r) = divide(&mut indices, data, dim);
            println!("dim = {}, boundary = {}", dim, boundary);
            println!("divided into ");
            println!("   L: {:?}", indices_l.iter().map(|&i| data[i]).collect::<Vec<Vector<D>>>());
            println!("   R: {:?}", indices_r.iter().map(|&i| data[i]).collect::<Vec<Vector<D>>>());
            println!("");

            boundaries.insert(node_index, boundary);
            let next_dim = (dim + 1) % D;
            stack.push((indices_l, node_index * 2 + 0, next_dim));
            stack.push((indices_r, node_index * 2 + 1, next_dim));
        }
        KdTree { data, leaf_size, boundaries, leaves }
    }

    pub fn print(&self) {
        print_tree::<D>(&self.boundaries, &self.leaves, &self.data);
    }

    pub fn search(&self, query: &Vector<D>) -> (Option<usize>, f64) {
        let leaf_index = find_leaf(query, &self.boundaries);
        let Some(indices) = self.leaves.get(&leaf_index) else {
            panic_leaf_node_not_found(query, leaf_index);
        };

        let (argmin, distance) = find_nearest(query, &indices, self.data);
        find_nearest_in_other_areas(query, &argmin, distance, leaf_index, &self.data, &self.boundaries, &self.leaves)
    }
}

fn squared_diff(a: f64, b: f64) -> f64 {
    (a - b) * (a - b)
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

fn find_leaf<const D: usize>(query: &Vector<D>, boundaries: &BTreeMap<usize, f64>) -> usize {
    let mut node_index = 1;
    let mut dim: usize = 0;
    while let Some(&boundary) = boundaries.get(&node_index) {
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
    boundaries: &BTreeMap<usize, f64>,
    leaves: &BTreeMap<usize, Vec<usize>>,
    data: &[Vector<D>],
) {
    let mut stack = Vec::from([(1, 0)]);
    while stack.len() != 0 {
        let (node_index, dim) = stack.pop().unwrap();

        let depth = f64::log2(node_index as f64) as usize;
        if let Some(indices) = leaves.get(&node_index) {
            info!("{} {:3}  {:?}", " ".repeat(2 * depth), node_index,
                indices.iter().map(|&i| data[i]).collect::<Vec<Vector<D>>>());
            continue;
        };

        let b = match boundaries.get(&node_index) {
            None => "".to_string(),
            Some(boundary) => format!("{:.5}", boundary),
        };
        info!("{} index = {:2}:  dim = {}:  boundary = {}", " ".repeat(2 * depth), node_index, dim, b);

        stack.push((node_index * 2 + 0, (dim + 1) % D));
        stack.push((node_index * 2 + 1, (dim + 1) % D));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_vecs(data: &[[f64; 2]]) -> Vec<Vector<2>> {
        data
            .iter()
            .map(|s| (*s).into())
            .collect::<Vec<Vector<2>>>()
    }

    fn search_and_print<const D: usize>(
        query: &Vector<D>,
        data: &[Vector<D>],
        tree: &KdTree<D>,
    ) {
        let (argmin, distance) = tree.search(query);
        match argmin {
            None => info!("Nearest element with query {:?} not found.", query),
            Some(i) => info!("query = {:?}, found = {:?}, distance = {}", query, data[i], distance),
        }
    }

    #[test]
    fn test_divide1() {
        let raw_data: [[f64; 2]; 10] = [
            [13., 10.],
            [16., 14.],
            [14., 11.],
            [11., 18.],
            [15., 12.],
            [10., 13.],
            [17., 15.],
            [12., 19.],
            [19., 18.],
            [18., 16.],
        ];

        let dim = 0;
        let data = to_vecs(&raw_data);
        let mut indices = (0..data.len()).collect();
        let (boundary, indices_l, indices_r) = divide(&mut indices, &data, dim);
        assert_eq!(boundary, 15.);
        for &i in indices_l.iter() {
            assert!(data[i][dim] < boundary);
        }
        for &i in indices_r.iter() {
            assert!(data[i][dim] >= boundary);
        }
    }

    #[test]
    fn test_divide2() {
        let raw_data: [[f64; 2]; 11] = [
            [13., 10.],
            [16., 14.],
            [14., 11.],
            [11., 18.],
            [15., 12.],
            [10., 13.],
            [17., 15.],
            [12., 19.],
            [19., 18.],
            [18., 16.],
            [20., 16.],
        ];

        let dim = 0;
        let data = to_vecs(&raw_data);
        let mut indices = (0..data.len()).collect();
        let (boundary, indices_l, indices_r) = divide(&mut indices, &data, dim);
        assert_eq!(boundary, 16.);
        for &i in indices_l.iter() {
            assert!(data[i][dim] < boundary);
        }
        for &i in indices_r.iter() {
            assert!(data[i][dim] >= boundary);
        }
    }

    #[test]
    fn test_divide3() {
        let raw_data: [[f64; 2]; 11] = [
            [13., 10.],
            [17., 14.],
            [15., 13.],
            [15., 15.],
            [15., 19.],
            [13., 11.],
            [13., 18.],
            [13., 12.],
            [17., 18.],
            [17., 16.],
            [28., 16.],
        ];

        let dim = 0;
        let data = to_vecs(&raw_data);
        let mut indices = (0..data.len()).collect();
        let (boundary, indices_l, indices_r) = divide(&mut indices, &data, dim);
        assert_eq!(boundary, 15.);
        for &i in indices_l.iter() {
            assert!(data[i][dim] < boundary);
        }
        for &i in indices_r.iter() {
            assert!(data[i][dim] >= boundary);
        }
    }

    #[test]
    fn test_find_leaf() {
        let raw_data: [[f64; 2]; 25] = [
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

        let vecs = to_vecs(&raw_data);
        let tree = KdTree::new(&vecs, 2);

        let leaf_index = find_leaf(&Vector::<2>::new(10., 15.), &tree.boundaries);
        assert_eq!(leaf_index, 22);

        let leaf_index = find_leaf(&Vector::<2>::new(11., 13.), &tree.boundaries);
        assert_eq!(leaf_index, 28);

        let leaf_index = find_leaf(&Vector::<2>::new(3., 2.), &tree.boundaries);
        assert_eq!(leaf_index, 16);

        let leaf_index = find_leaf(&Vector::<2>::new(8., 17.), &tree.boundaries);
        assert_eq!(leaf_index, 22);
    }

    #[test]
    fn test_search_leaf_size_2() {
        let raw_data: [[f64; 2]; 25] = [
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

        let vecs = to_vecs(&raw_data);

        let leaf_size = 2;

        let tree = KdTree::new(&vecs, leaf_size);
        tree.print();
        // for query in data.iter() {
        //     search_and_print(query, data, &tree);
        // }

        search_and_print(&Vector::<2>::new(10., 15.), &vecs, &tree);
        // search_and_print(&Vector::<2>::new(14., 13.), &boundaries, &leaves, data);
        // search_and_print(&Vector::<2>::new(10., 11.), &boundaries, &leaves, data);
        // search_and_print(&Vector::<2>::new(2., 11.), &boundaries, &leaves, data);
    }

    #[test]
    fn test_search_leaf_size_1() {
        let raw_data: [[f64; 2]; 25] = [
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

        let raw_data: [[f64; 2]; 10] = [
            [-4., 5.],
            [-3., -5.],
            [-3., -3.],
            [-3., 2.],
            [1., 1.],
            [1., 3.],
            [2., -2.],
            [3., 2.],
            [3., 4.],
            [5., -2.],
        ];

        let vecs = to_vecs(&raw_data);

        let leaf_size = 1;

        let tree = KdTree::new(&vecs, leaf_size);
        tree.print();
        // for query in data.iter() {
        //     search_and_print(query, data, &tree);
        // }

        println!("tree.leaves:");
        println!("{:?}", tree.leaves);
        search_and_print(&Vector::<2>::new(10., 15.), &vecs, &tree);
        // search_and_print(&Vector::<2>::new(14., 13.), &boundaries, &leaves, data);
        // search_and_print(&Vector::<2>::new(10., 11.), &boundaries, &leaves, data);
        // search_and_print(&Vector::<2>::new(2., 11.), &boundaries, &leaves, data);
    }

    #[test]
    fn test_new_with_duplicated_elements() {
        let raw_data: [[f64; 2]; 15] = [
            [-3., -1.],
            [-3., -1.],
            [-3., -1.],
            [-3., -1.],
            [-3., -1.],
            [-3., 3.],
            [-3., -3.],
            [-1., -1.],
            [-1., 3.],
            [1., 3.],
            [2., -3.],
            [2., -1.],
            [2., 3.],
            [3., -1.],
            [4., 1.],
        ];
        let leaf_size = 1;
        let vecs = to_vecs(&raw_data);
        let tree = KdTree::new(&vecs, leaf_size);
        tree.print();
        search_and_print(&Vector::<2>::new(0., -4.), &vecs, &tree);
    }
    #[test]
    fn test_find_nearest_in_other_areas() {
        let raw_data: [[f64; 2]; 11] = [
            [-3., -1.],
            [-3., 3.],
            [-3., -3.],
            [-1., -1.],
            [-1., 3.],
            [1., 3.],
            [2., -3.],
            [2., -1.],
            [2., 3.],
            [3., -1.],
            [4., 1.],
        ];
        let leaf_size = 2;
        let vecs = to_vecs(&raw_data);
        let tree = KdTree::new(&vecs, leaf_size);
        tree.print();
        search_and_print(&Vector::<2>::new(0., -4.), &vecs, &tree);
    }

    #[test]
    fn test_find_dim() {
        assert_eq!(find_dim::<4>(1), 0);
        assert_eq!(find_dim::<4>(2), 1);
        assert_eq!(find_dim::<4>(3), 1);
        assert_eq!(find_dim::<4>(4), 2);
        assert_eq!(find_dim::<4>(5), 2);
        assert_eq!(find_dim::<4>(10), 3);
        assert_eq!(find_dim::<4>(11), 3);
        assert_eq!(find_dim::<4>(18), 0);
        assert_eq!(find_dim::<4>(52), 1);
        assert_eq!(find_dim::<4>(63), 1);
        assert_eq!(find_dim::<4>(64), 2);
    }
}
