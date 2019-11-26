/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap;

// # Author: Leland McInnes <leland.mcinnes@gmail.com>
// #
// # License: BSD 3 clause
// from __future__ import print_function
// from collections import deque, namedtuple
// from warnings import warn

// import numpy as np
// import numba

// from umap.sparse import sparse_mul, sparse_diff, sparse_sum

// from umap.utils import tau_rand_int, norm

// import scipy.sparse
// import locale

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class RpTree {

// locale.setlocale(locale.LC_NUMERIC, "C")

  // # Used for a floating point "nearly zero" comparison
  private static final float EPS = 1e-8F;

// RandomProjectionTreeNode = namedtuple(
//     "RandomProjectionTreeNode",
//     ["indices", "is_leaf", "hyperplane", "offset", "left_child", "right_child"],
// )

// FlatTree = namedtuple("FlatTree", ["hyperplanes", "offsets", "children", "indices"])


// @numba.njit(fastmath=true)
// def angular_random_projection_split(data, indices, rng_state):
//     """Given a set of ``indices`` for data points from ``data``, create
//     a random hyperplane to split the data, returning two arrays indices
//     that fall on either side of the hyperplane. This is the basis for a
//     random projection tree, which simply uses this splitting recursively.
//     This particular split uses cosine distance to determine the hyperplane
//     and which side each data sample falls on.
//     Parameters
//     ----------
//     data: array of shape (n_samples, n_features)
//         The original data to be split
//     indices: array of shape (tree_node_size,)
//         The indices of the elements in the ``data`` array that are to
//         be split in the current operation.
//     rng_state: array of int64, shape (3,)
//         The internal state of the rng
//     Returns
//     -------
//     indices_left: array
//         The elements of ``indices`` that fall on the "left" side of the
//         random hyperplane.
//     indices_right: array
//         The elements of ``indices`` that fall on the "left" side of the
//         random hyperplane.
//     """
//     dim = data.shape[1]

//     # Select two random points, set the hyperplane between them
//     left_index = tau_rand_int(rng_state) % indices.shape[0]
//     right_index = tau_rand_int(rng_state) % indices.shape[0]
//     right_index += left_index == right_index
//     right_index = right_index % indices.shape[0]
//     left = indices[left_index]
//     right = indices[right_index]

//     left_norm = norm(data[left])
//     right_norm = norm(data[right])

//     if abs(left_norm) < EPS:
//         left_norm = 1.0

//     if abs(right_norm) < EPS:
//         right_norm = 1.0

//     # Compute the normal vector to the hyperplane (the vector between
//     # the two points)
//     hyperplane_vector = np.empty(dim, dtype=np.float32)

//     for d in range(dim):
//         hyperplane_vector[d] = (data[left, d] / left_norm) - (
//             data[right, d] / right_norm
//         )

//     hyperplane_norm = norm(hyperplane_vector)
//     if abs(hyperplane_norm) < EPS:
//         hyperplane_norm = 1.0

//     for d in range(dim):
//         hyperplane_vector[d] = hyperplane_vector[d] / hyperplane_norm

//     # For each point compute the margin (project into normal vector)
//     # If we are on lower side of the hyperplane put in one pile, otherwise
//     # put it in the other pile (if we hit hyperplane on the nose, flip a coin)
//     n_left = 0
//     n_right = 0
//     side = np.empty(indices.shape[0], np.int8)
//     for i in range(indices.shape[0]):
//         margin = 0.0
//         for d in range(dim):
//             margin += hyperplane_vector[d] * data[indices[i], d]

//         if abs(margin) < EPS:
//             side[i] = tau_rand_int(rng_state) % 2
//             if side[i] == 0:
//                 n_left += 1
//             else:
//                 n_right += 1
//         else if margin > 0:
//             side[i] = 0
//             n_left += 1
//         else:
//             side[i] = 1
//             n_right += 1

//     # Now that we have the counts allocate arrays
//     indices_left = np.empty(n_left, dtype=np.int64)
//     indices_right = np.empty(n_right, dtype=np.int64)

//     # Populate the arrays with indices according to which side they fell on
//     n_left = 0
//     n_right = 0
//     for i in range(side.shape[0]):
//         if side[i] == 0:
//             indices_left[n_left] = indices[i]
//             n_left += 1
//         else:
//             indices_right[n_right] = indices[i]
//             n_right += 1

//     return indices_left, indices_right, hyperplane_vector, null


  //     Given a set of ``indices`` for data points from ``data``, create
//     a random hyperplane to split the data, returning two arrays indices
//     that fall on either side of the hyperplane. This is the basis for a
//     random projection tree, which simply uses this splitting recursively.
//     This particular split uses euclidean distance to determine the hyperplane
//     and which side each data sample falls on.
//     Parameters
//     ----------
//     data: array of shape (n_samples, n_features)
//         The original data to be split
//     indices: array of shape (tree_node_size,)
//         The indices of the elements in the ``data`` array that are to
//         be split in the current operation.
//     rng_state: array of int64, shape (3,)
//         The internal state of the rng
//     Returns
//     -------
//     indices_left: array
//         The elements of ``indices`` that fall on the "left" side of the
//         random hyperplane.
//     indices_right: array
//         The elements of ``indices`` that fall on the "left" side of the
//         random hyperplane.
  static Object[] euclidean_random_projection_split(final Matrix data, final int[] indices, final long[] rng_state) {
    final int dim = data.cols();

    // Select two random points, set the hyperplane between them
    final int left_index = Math.abs(Utils.tau_rand_int(rng_state) % indices.length);
    int right_index = Math.abs(Utils.tau_rand_int(rng_state) % indices.length);
    right_index += left_index == right_index ? 1 : 0;
    right_index = right_index % indices.length;
    int left = indices[left_index];
    int right = indices[right_index];

    // Compute the normal vector to the hyperplane (the vector between
    // the two points) and the offset from the origin
    float hyperplane_offset = 0.0F;
    float[] hyperplane_vector = new float[dim];

    for (int d = 0; d < dim; ++d) {
      hyperplane_vector[d] = data.get(left, d) - data.get(right, d);
      hyperplane_offset -= hyperplane_vector[d] * (data.get(left, d) + data.get(right, d)) / 2.0;
    }

    // For each point compute the margin (project into normal vector, add offset)
    // If we are on lower side of the hyperplane put in one pile, otherwise
    // put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    int n_left = 0;
    int n_right = 0;
    //side = np.empty(indices.shape[0], np.int8);
    boolean[] side = new boolean[indices.length];
    for (int i = 0; i < indices.length; ++i) {
      float margin = hyperplane_offset;
      for (int d = 0; d < dim; ++d) {
        margin += hyperplane_vector[d] * data.get(indices[i], d);
      }
      if (Math.abs(margin) < EPS) {
        side[i] = Utils.tau_rand_int(rng_state) % 2 == 0;
        if (!side[i]) {
          ++n_left;
        } else {
          ++n_right;
        }
      } else if (margin > 0) {
        side[i] = false;
        ++n_left;
      } else {
        side[i] = true;
        ++n_right;
      }
    }
    // Now that we have the counts allocate arrays
    final int[] indices_left = new int[n_left];
    final int[] indices_right = new int[n_right];

    // Populate the arrays with indices according to which side they fell on
    n_left = 0;
    n_right = 0;
    for (int i = 0; i < side.length; ++i) {
      if (!side[i]) {
        indices_left[n_left++] = indices[i];
      } else {
        indices_right[n_right++] = indices[i];
      }
    }
    return new Object[]{indices_left, indices_right, hyperplane_vector, hyperplane_offset};
  }


// @numba.njit(fastmath=true)
// def sparse_angular_random_projection_split(inds, indptr, data, indices, rng_state):
//     """Given a set of ``indices`` for data points from a sparse data set
//     presented in csr sparse format as inds, indptr and data, create
//     a random hyperplane to split the data, returning two arrays indices
//     that fall on either side of the hyperplane. This is the basis for a
//     random projection tree, which simply uses this splitting recursively.
//     This particular split uses cosine distance to determine the hyperplane
//     and which side each data sample falls on.
//     Parameters
//     ----------
//     inds: array
//         CSR format index array of the matrix
//     indptr: array
//         CSR format index pointer array of the matrix
//     data: array
//         CSR format data array of the matrix
//     indices: array of shape (tree_node_size,)
//         The indices of the elements in the ``data`` array that are to
//         be split in the current operation.
//     rng_state: array of int64, shape (3,)
//         The internal state of the rng
//     Returns
//     -------
//     indices_left: array
//         The elements of ``indices`` that fall on the "left" side of the
//         random hyperplane.
//     indices_right: array
//         The elements of ``indices`` that fall on the "left" side of the
//         random hyperplane.
//     """
//     # Select two random points, set the hyperplane between them
//     left_index = tau_rand_int(rng_state) % indices.shape[0]
//     right_index = tau_rand_int(rng_state) % indices.shape[0]
//     right_index += left_index == right_index
//     right_index = right_index % indices.shape[0]
//     left = indices[left_index]
//     right = indices[right_index]

//     left_inds = inds[indptr[left] : indptr[left + 1]]
//     left_data = data[indptr[left] : indptr[left + 1]]
//     right_inds = inds[indptr[right] : indptr[right + 1]]
//     right_data = data[indptr[right] : indptr[right + 1]]

//     left_norm = norm(left_data)
//     right_norm = norm(right_data)

//     if abs(left_norm) < EPS:
//         left_norm = 1.0

//     if abs(right_norm) < EPS:
//         right_norm = 1.0

//     # Compute the normal vector to the hyperplane (the vector between
//     # the two points)
//     normalized_left_data = left_data / left_norm
//     normalized_right_data = right_data / right_norm
//     hyperplane_inds, hyperplane_data = sparse_diff(
//         left_inds, normalized_left_data, right_inds, normalized_right_data
//     )

//     hyperplane_norm = norm(hyperplane_data)
//     if abs(hyperplane_norm) < EPS:
//         hyperplane_norm = 1.0
//     for d in range(hyperplane_data.shape[0]):
//         hyperplane_data[d] = hyperplane_data[d] / hyperplane_norm

//     # For each point compute the margin (project into normal vector)
//     # If we are on lower side of the hyperplane put in one pile, otherwise
//     # put it in the other pile (if we hit hyperplane on the nose, flip a coin)
//     n_left = 0
//     n_right = 0
//     side = np.empty(indices.shape[0], np.int8)
//     for i in range(indices.shape[0]):
//         margin = 0.0

//         i_inds = inds[indptr[indices[i]] : indptr[indices[i] + 1]]
//         i_data = data[indptr[indices[i]] : indptr[indices[i] + 1]]

//         mul_inds, mul_data = sparse_mul(
//             hyperplane_inds, hyperplane_data, i_inds, i_data
//         )
//         for d in range(mul_data.shape[0]):
//             margin += mul_data[d]

//         if abs(margin) < EPS:
//             side[i] = tau_rand_int(rng_state) % 2
//             if side[i] == 0:
//                 n_left += 1
//             else:
//                 n_right += 1
//         else if margin > 0:
//             side[i] = 0
//             n_left += 1
//         else:
//             side[i] = 1
//             n_right += 1

//     # Now that we have the counts allocate arrays
//     indices_left = np.empty(n_left, dtype=np.int64)
//     indices_right = np.empty(n_right, dtype=np.int64)

//     # Populate the arrays with indices according to which side they fell on
//     n_left = 0
//     n_right = 0
//     for i in range(side.shape[0]):
//         if side[i] == 0:
//             indices_left[n_left] = indices[i]
//             n_left += 1
//         else:
//             indices_right[n_right] = indices[i]
//             n_right += 1

//     hyperplane = np.vstack((hyperplane_inds, hyperplane_data))

//     return indices_left, indices_right, hyperplane, null


  //     Given a set of ``indices`` for data points from a sparse data set
//     presented in csr sparse format as inds, indptr and data, create
//     a random hyperplane to split the data, returning two arrays indices
//     that fall on either side of the hyperplane. This is the basis for a
//     random projection tree, which simply uses this splitting recursively.
//     This particular split uses cosine distance to determine the hyperplane
//     and which side each data sample falls on.
//     Parameters
//     ----------
//     inds: array
//         CSR format index array of the matrix
//     indptr: array
//         CSR format index pointer array of the matrix
//     data: array
//         CSR format data array of the matrix
//     indices: array of shape (tree_node_size,)
//         The indices of the elements in the ``data`` array that are to
//         be split in the current operation.
//     rng_state: array of int64, shape (3,)
//         The internal state of the rng
//     Returns
//     -------
//     indices_left: array
//         The elements of ``indices`` that fall on the "left" side of the
//         random hyperplane.
//     indices_right: array
//         The elements of ``indices`` that fall on the "left" side of the
//         random hyperplane.
  static Object[] sparse_euclidean_random_projection_split(final int[] inds, final int[] indptr, final float[] data, final int[] indices, final long[] rng_state) {
    // Select two random points, set the hyperplane between them
    int left_index = Utils.tau_rand_int(rng_state) % indices.length;
    int right_index = Utils.tau_rand_int(rng_state) % indices.length;
    right_index += left_index == right_index ? 1 : 0;
    right_index = right_index % indices.length;
    int left = indices[left_index];
    int right = indices[right_index];

    int[] left_inds = MathUtils.subarray(inds, indptr[left], indptr[left + 1]);
    float[] left_data = MathUtils.subarray(data, indptr[left], indptr[left + 1]);
    int[] right_inds = MathUtils.subarray(inds, indptr[right], indptr[right + 1]);
    float[] right_data = MathUtils.subarray(data, indptr[right], indptr[right + 1]);

    // Compute the normal vector to the hyperplane (the vector between
    // the two points) and the offset from the origin
    float hyperplane_offset = 0.0F;
    final Object[] sd = Sparse.sparse_diff(left_inds, left_data, right_inds, right_data);
    final int[] hyperplane_inds = (int[]) sd[0];
    final float[] hyperplane_data = (float[]) sd[1];
    final Object[] ss = Sparse.sparse_sum(left_inds, left_data, right_inds, right_data);
    int[] offset_inds = (int[]) ss[0];
    float[] offset_data = MathUtils.divide((float[]) ss[1], 2.0F);
    final Object[] sm = Sparse.sparse_mul(hyperplane_inds, hyperplane_data, offset_inds, offset_data);
    offset_inds = (int[]) sm[0];
    offset_data = (float[]) sm[1];

    for (int d = 0; d < offset_data.length; ++d) {
      hyperplane_offset -= offset_data[d];
    }

    // For each point compute the margin (project into normal vector, add offset)
    // If we are on lower side of the hyperplane put in one pile, otherwise
    // put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    int n_left = 0;
    int n_right = 0;
    boolean[] side = new boolean[indices.length];
    for (int i = 0; i < indices.length; ++i) {
      float margin = hyperplane_offset;
      int[] i_inds = MathUtils.subarray(inds, indptr[indices[i]], indptr[indices[i] + 1]);
      float[] i_data = MathUtils.subarray(data, indptr[indices[i]], indptr[indices[i] + 1]);

      final Object[] spm = Sparse.sparse_mul(hyperplane_inds, hyperplane_data, i_inds, i_data);
      final int[] mul_inds = (int[]) spm[0];
      final float[] mul_data = (float[]) spm[1];
      for (int d = 0; d < mul_data.length; ++d) {
        margin += mul_data[d];
      }

      if (Math.abs(margin) < EPS) {
        side[i] = Utils.tau_rand_int(rng_state) % 2 == 0;
        if (!side[i]) {
          n_left += 1;
        } else {
          n_right += 1;
        }
      } else if (margin > 0) {
        side[i] = false;
        n_left += 1;
      } else {
        side[i] = true;
        n_right += 1;
      }
    }

    // Now that we have the counts allocate arrays
    int[] indices_left = new int[n_left];
    int[] indices_right = new int[n_right];

    // Populate the arrays with indices according to which side they fell on
    n_left = 0;
    n_right = 0;
    for (int i = 0; i < side.length; ++i) {
      if (!side[i]) {
        indices_left[n_left] = indices[i];
        n_left += 1;
      } else {
        indices_right[n_right] = indices[i];
        n_right += 1;
      }
    }

    final Hyperplane hyperplane = new Hyperplane(hyperplane_inds, hyperplane_data);

    return new Object[]{indices_left, indices_right, hyperplane, hyperplane_offset};
  }


  static RandomProjectionTreeNode make_euclidean_tree(final Matrix data, final int[] indices, final long[] rng_state, final int leaf_size) {
    if (indices.length > leaf_size) {
      final Object[] erps = euclidean_random_projection_split(data, indices, rng_state);
      final int[] left_indices = (int[]) erps[0];
      final int[] right_indices = (int[]) erps[1];
      final Hyperplane hyperplane = new Hyperplane((float[]) erps[2]);
      final float offset = (float) erps[3];

      final RandomProjectionTreeNode left_node = make_euclidean_tree(data, left_indices, rng_state, leaf_size);
      final RandomProjectionTreeNode right_node = make_euclidean_tree(data, right_indices, rng_state, leaf_size);

      return new RandomProjectionTreeNode(null, false, hyperplane, offset, left_node, right_node);
    } else {
      return new RandomProjectionTreeNode(indices, true, null, null, null, null);
    }
  }

// def make_angular_tree(data, indices, rng_state, leaf_size=30):
//     if indices.shape[0] > leaf_size:
//         left_indices, right_indices, hyperplane, offset = angular_random_projection_split(
//             data, indices, rng_state
//         )

//         left_node = make_angular_tree(data, left_indices, rng_state, leaf_size)
//         right_node = make_angular_tree(data, right_indices, rng_state, leaf_size)

//         node = RandomProjectionTreeNode(
//             null, false, hyperplane, offset, left_node, right_node
//         )
//     else:
//         node = RandomProjectionTreeNode(indices, true, null, null, null, null)

//     return node


  static RandomProjectionTreeNode make_sparse_euclidean_tree(final int[] inds, final int[] indptr, final float[] data, final int[] indices, final long[] rng_state, final int leaf_size) {
    if (indices.length > leaf_size) {
      final Object[] erps = sparse_euclidean_random_projection_split(inds, indptr, data, indices, rng_state);
      final int[] left_indices = (int[]) erps[0];
      final int[] right_indices = (int[]) erps[1];
      final Hyperplane hyperplane = (Hyperplane) erps[2];
      final float offset = (float) erps[3];

      RandomProjectionTreeNode left_node = make_sparse_euclidean_tree(inds, indptr, data, left_indices, rng_state, leaf_size);
      RandomProjectionTreeNode right_node = make_sparse_euclidean_tree(inds, indptr, data, right_indices, rng_state, leaf_size);

      return new RandomProjectionTreeNode(null, false, hyperplane, offset, left_node, right_node);
    } else {
      return new RandomProjectionTreeNode(indices, true, null, null, null, null);
    }
  }


// def make_sparse_angular_tree(inds, indptr, data, indices, rng_state, leaf_size=30):
//     if indices.shape[0] > leaf_size:
//         left_indices, right_indices, hyperplane, offset = sparse_angular_random_projection_split(
//             inds, indptr, data, indices, rng_state
//         )

//         left_node = make_sparse_angular_tree(
//             inds, indptr, data, left_indices, rng_state, leaf_size
//         )
//         right_node = make_sparse_angular_tree(
//             inds, indptr, data, right_indices, rng_state, leaf_size
//         )

//         node = RandomProjectionTreeNode(
//             null, false, hyperplane, offset, left_node, right_node
//         )
//     else:
//         node = RandomProjectionTreeNode(indices, true, null, null, null, null)

//     return node


  //     """Construct a random projection tree based on ``data`` with leaves
//     of size at most ``leaf_size``.
//     Parameters
//     ----------
//     data: array of shape (n_samples, n_features)
//         The original data to be split
//     rng_state: array of int64, shape (3,)
//         The internal state of the rng
//     leaf_size: int (optional, default 30)
//         The maximum size of any leaf node in the tree. Any node in the tree
//         with more than ``leaf_size`` will be split further to create child
//         nodes.
//     angular: bool (optional, default false)
//         Whether to use cosine/angular distance to create splits in the tree,
//         || euclidean distance.
//     Returns
//     -------
//     node: RandomProjectionTreeNode
//         A random projection tree node which links to its child nodes. This
//         provides the full tree below the returned node.
//     """
  static RandomProjectionTreeNode make_tree(final Matrix data, final long[] rng_state, final int leaf_size, final boolean angular) {
    final boolean is_sparse = data instanceof CsrMatrix;
    //final int indices = np.arange(data.shape[0]);
    final int[] indices = MathUtils.identity(data.rows());

    // Make a tree recursively until we get below the leaf size
    if (is_sparse) {
      final CsrMatrix csrData = (CsrMatrix) data;
      final int[] inds = csrData.indices;
      final int[] indptr = csrData.indptr;
      final float[] spdata = csrData.data;

      if (angular) {
        throw new UnsupportedOperationException();
        //return make_sparse_angular_tree(inds, indptr, spdata, indices, rng_state, leaf_size);
      } else {
        return make_sparse_euclidean_tree(inds, indptr, spdata, indices, rng_state, leaf_size);
      }
    } else {
      if (angular) {
        throw new UnsupportedOperationException();
        //return make_angular_tree(data, indices, rng_state, leaf_size);
      } else {
        return make_euclidean_tree(data, indices, rng_state, leaf_size);
      }
    }
  }


// def num_nodes(tree):
//     """Determine the number of nodes in a tree"""
//     if tree.is_leaf:
//         return 1
//     else:
//         return 1 + num_nodes(tree.left_child) + num_nodes(tree.right_child)


// def num_leaves(tree):
//     """Determine the number of leaves in a tree"""
//     if tree.is_leaf:
//         return 1
//     else:
//         return num_leaves(tree.left_child) + num_leaves(tree.right_child)


  //     """Determine the most number on non zeros in a hyperplane entry"""
  static int max_sparse_hyperplane_size(RandomProjectionTreeNode tree) {
    if (tree.isLeaf()) {
      return 0;
    } else {
      return Math.max(
        tree.getHyperplane().shape[1],
        Math.max(max_sparse_hyperplane_size(tree.getLeftChild()),
          max_sparse_hyperplane_size(tree.getRightChild()))
      );
    }
  }


  static int[] recursive_flatten(final RandomProjectionTreeNode tree, final float[][][] hyperplanes, final float[] offsets, final int[][] children, final int[][] indices, int node_num, int leaf_num) {
    if (tree.isLeaf()) {
      children[node_num][0] = -leaf_num;
      //indices[leaf_num, :tree.getIndices().shape[0]] =tree.getIndices();
      indices[leaf_num] = tree.getIndices();
      leaf_num += 1;
      return new int[]{node_num, leaf_num};
    } else {
      if (tree.getHyperplane().shape.length > 1) {
        // sparse case
        //hyperplanes[node_num][:, :tree.getHyperplane().shape[1]] =tree.getHyperplane();
        throw new UnsupportedOperationException();
      } else {
        hyperplanes[node_num] = null; //tree.getHyperplane().data; // todo
      }
      offsets[node_num] = tree.getOffset();
      children[node_num][0] = node_num + 1;
      final int old_node_num = node_num;
      final int[] t = recursive_flatten(tree.getLeftChild(), hyperplanes, offsets, children, indices, node_num + 1, leaf_num);
      node_num = t[0];
      leaf_num = t[1];
      children[old_node_num][1] = node_num + 1;
      return recursive_flatten(tree.getRightChild(), hyperplanes, offsets, children, indices, node_num + 1, leaf_num);
    }
  }

  private static int[][] negOnes(final int a, final int b) {
    final int[][] res = new int[a][b];
    for (final int[] row : res) {
      Arrays.fill(row, -1);
    }
    return res;
  }


  static FlatTree flatten_tree(final RandomProjectionTreeNode tree, int leaf_size) {
    final int n_nodes = tree.num_nodes();
    final int n_leaves = tree.num_leaves();

    final float[][][] hyperplanes;
    if (tree.getHyperplane().shape.length > 1) {
      // sparse case
      final int max_hyperplane_nnz = max_sparse_hyperplane_size(tree);
      hyperplanes = new float[n_nodes][tree.getHyperplane().shape[0]][max_hyperplane_nnz];
    } else {
      hyperplanes = new float[n_nodes][tree.getHyperplane().shape[0]][1]; // todo ???
    }
    final float[] offsets = new float[n_nodes];
    final int[][] children = negOnes(n_nodes, 2);
    final int[][] indices = negOnes(n_leaves, leaf_size);
    recursive_flatten(tree, hyperplanes, offsets, children, indices, 0, 0);
    return new FlatTree(hyperplanes, offsets, children, indices);
  }


 static int select_side(final float[] hyperplane, final float offset, final float[] point, final long[] rng_state) {
   float margin = offset;
   for (int d = 0; d < point.length; ++d) {
     margin += hyperplane[d] * point[d];
   }

   if (Math.abs(margin) < EPS) {
     final int side = Math.abs(Utils.tau_rand_int(rng_state) % 2);
     if (side == 0) {
       return 0;
     } else {
       return 1;
     }
   } else if (margin > 0) {
     return 0;
   } else {
     return 1;
   }
 }

 static int[] search_flat_tree(final float[] point, final float[][] hyperplanes, final float[] offsets, final int[][] children, final int[][] indices, final long[] rng_state) {
   int node = 0;
   while (children[node][0] > 0) {
     final int side = select_side(hyperplanes[node], offsets[node], point, rng_state);
     if (side == 0) {
       node = children[node][0];
     } else {
       node = children[node][1];
     }
   }

   return indices[-children[node][0]];
 }


  //     """Build a random projection forest with ``n_trees``.
//
//     Parameters
//     ----------
//     data
//     n_neighbors
//     n_trees
//     rng_state
//     angular
//
//     Returns
//     -------
//     forest: list
//         A list of random projection trees.
//     """
  static List<FlatTree> make_forest(final Matrix data, final int n_neighbors, final int n_trees, final long[] rng_state, final boolean angular) {
    final ArrayList<FlatTree> result = new ArrayList<>();
    final int leaf_size = Math.max(10, n_neighbors);
    try {
      for (int i = 0; i < n_trees; ++i) {
        result.add(flatten_tree(make_tree(data, rng_state, leaf_size, angular), leaf_size));
      }
    } catch (RuntimeException e) {
      e.printStackTrace();
      Utils.message("Random Projection forest initialisation failed due to recursion limit being reached. Something is a little strange with your data, and this may take longer than normal to compute.");
    }
    return result;
  }


  //     Generate an array of sets of candidate nearest neighbors by
//     constructing a random projection forest and taking the leaves of all the
//     trees. Any given tree has leaves that are a set of potential nearest
//     neighbors. Given enough trees the set of all such leaves gives a good
//     likelihood of getting a good set of nearest neighbors in composite. Since
//     such a random projection forest is inexpensive to compute, this can be a
//     useful means of seeding other nearest neighbor algorithms.
//     Parameters
//     ----------
//     data: array of shape (n_samples, n_features)
//         The data for which to generate nearest neighbor approximations.
//     n_neighbors: int
//         The number of nearest neighbors to attempt to approximate.
//     rng_state: array of int64, shape (3,)
//         The internal state of the rng
//     n_trees: int (optional, default 10)
//         The number of trees to build in the forest construction.
//     angular: bool (optional, default false)
//         Whether to use angular/cosine distance for random projection tree
//         construction.
//     Returns
//     -------
//     leaf_array: array of shape (n_leaves, max(10, n_neighbors))
//         Each row of leaf array is a list of indices found in a given leaf.
//         Since not all leaves are the same size the arrays are padded out with -1
//         to ensure we can return a single ndarray.
  static int[][] rptree_leaf_array(final List<FlatTree> rpForest) {
    if (rpForest.size() > 0) {
      final int[][] leafArray = new int[rpForest.size()][];
      for (int k = 0; k < leafArray.length; ++k) {
        //leafArray[k] = rpForest.get(k).getIndices();
        leafArray[k] = rpForest.get(k).getIndices()[0]; // todo !!! datatype mismatch ???
      }
      return leafArray;
    } else {
      return new int[0][0]; // todo ingored -1 padding?
    }
  }
}
