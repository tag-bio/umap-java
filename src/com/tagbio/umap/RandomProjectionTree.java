/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Random projection trees.
 */
class RandomProjectionTree {

  // Used for a floating point "nearly zero" comparison
  private static final float EPS = 1e-8F;

// def angular_random_projection_split(data, indices, random):
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
//     random: array of int64, shape (3,)
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
//     left_index = tau_rand_int(random) % indices.shape[0]
//     right_index = tau_rand_int(random) % indices.shape[0]
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
//             side[i] = tau_rand_int(random) % 2
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
//     random: array of int64, shape (3,)
//         The internal state of the rng
//     Returns
//     -------
//     indices_left: array
//         The elements of ``indices`` that fall on the "left" side of the
//         random hyperplane.
//     indices_right: array
//         The elements of ``indices`` that fall on the "left" side of the
//         random hyperplane.
  static Object[] euclideanRandomProjectionSplit(final Matrix data, final int[] indices, final Random random) {
    final int dim = data.cols();

    // Select two random points, set the hyperplane between them
    final int leftIndex = random.nextInt(indices.length);
    int rightIndex = random.nextInt(indices.length);
    rightIndex += leftIndex == rightIndex ? 1 : 0;
    rightIndex = rightIndex % indices.length;
    int left = indices[leftIndex];
    int right = indices[rightIndex];

    // Compute the normal vector to the hyperplane (the vector between
    // the two points) and the offset from the origin
    float hyperplaneOffset = 0.0F;
    float[] hyperplaneVector = new float[dim];

    for (int d = 0; d < dim; ++d) {
      hyperplaneVector[d] = data.get(left, d) - data.get(right, d);
      hyperplaneOffset -= hyperplaneVector[d] * (data.get(left, d) + data.get(right, d)) / 2.0;
    }

    // For each point compute the margin (project into normal vector, add offset)
    // If we are on lower side of the hyperplane put in one pile, otherwise
    // put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    int nLeft = 0;
    int nRight = 0;
    //side = np.empty(indices.shape[0], np.int8);
    boolean[] side = new boolean[indices.length];
    for (int i = 0; i < indices.length; ++i) {
      float margin = hyperplaneOffset;
      for (int d = 0; d < dim; ++d) {
        margin += hyperplaneVector[d] * data.get(indices[i], d);
      }
      if (Math.abs(margin) < EPS) {
        side[i] = random.nextBoolean();
        if (!side[i]) {
          ++nLeft;
        } else {
          ++nRight;
        }
      } else if (margin > 0) {
        side[i] = false;
        ++nLeft;
      } else {
        side[i] = true;
        ++nRight;
      }
    }
    // Now that we have the counts allocate arrays
    final int[] indicesLeft = new int[nLeft];
    final int[] indicesRight = new int[nRight];

    // Populate the arrays with indices according to which side they fell on
    nLeft = 0;
    nRight = 0;
    for (int i = 0; i < side.length; ++i) {
      if (!side[i]) {
        indicesLeft[nLeft++] = indices[i];
      } else {
        indicesRight[nRight++] = indices[i];
      }
    }
    return new Object[]{indicesLeft, indicesRight, hyperplaneVector, hyperplaneOffset};
  }


// @numba.njit(fastmath=true)
// def sparse_angular_random_projection_split(inds, indptr, data, indices, random):
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
//     random: array of int64, shape (3,)
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
//     left_index = tau_rand_int(random) % indices.shape[0]
//     right_index = tau_rand_int(random) % indices.shape[0]
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
//             side[i] = tau_rand_int(random) % 2
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
//     random: array of int64, shape (3,)
//         The internal state of the rng
//     Returns
//     -------
//     indices_left: array
//         The elements of ``indices`` that fall on the "left" side of the
//         random hyperplane.
//     indices_right: array
//         The elements of ``indices`` that fall on the "left" side of the
//         random hyperplane.
  static Object[] sparseEuclideanRandomProjectionSplit(final int[] inds, final int[] indptr, final float[] data, final int[] indices, final Random random) {
    // Select two random points, set the hyperplane between them
    int leftIndex = random.nextInt(indices.length);
    int rightIndex = random.nextInt(indices.length);
    rightIndex += leftIndex == rightIndex ? 1 : 0;
    rightIndex = rightIndex % indices.length;
    int left = indices[leftIndex];
    int right = indices[rightIndex];

    int[] leftInds = MathUtils.subarray(inds, indptr[left], indptr[left + 1]);
    float[] leftData = MathUtils.subarray(data, indptr[left], indptr[left + 1]);
    int[] rightInds = MathUtils.subarray(inds, indptr[right], indptr[right + 1]);
    float[] rightData = MathUtils.subarray(data, indptr[right], indptr[right + 1]);

    // Compute the normal vector to the hyperplane (the vector between
    // the two points) and the offset from the origin
    float hyperplaneOffset = 0.0F;
    final Object[] sd = Sparse.sparseDiff(leftInds, leftData, rightInds, rightData);
    final int[] hyperplaneInds = (int[]) sd[0];
    final float[] hyperplaneData = (float[]) sd[1];
    final Object[] ss = Sparse.sparseSum(leftInds, leftData, rightInds, rightData);
    int[] offsetInds = (int[]) ss[0];
    float[] offsetData = MathUtils.divide((float[]) ss[1], 2.0F);
    final Object[] sm = Sparse.sparse_mul(hyperplaneInds, hyperplaneData, offsetInds, offsetData);
    offsetInds = (int[]) sm[0];
    offsetData = (float[]) sm[1];

    for (int d = 0; d < offsetData.length; ++d) {
      hyperplaneOffset -= offsetData[d];
    }

    // For each point compute the margin (project into normal vector, add offset)
    // If we are on lower side of the hyperplane put in one pile, otherwise
    // put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    int nLeft = 0;
    int nRight = 0;
    boolean[] side = new boolean[indices.length];
    for (int i = 0; i < indices.length; ++i) {
      float margin = hyperplaneOffset;
      int[] iInds = MathUtils.subarray(inds, indptr[indices[i]], indptr[indices[i] + 1]);
      float[] iData = MathUtils.subarray(data, indptr[indices[i]], indptr[indices[i] + 1]);

      final Object[] spm = Sparse.sparse_mul(hyperplaneInds, hyperplaneData, iInds, iData);
      final int[] mulInds = (int[]) spm[0];
      final float[] mulData = (float[]) spm[1];
      for (int d = 0; d < mulData.length; ++d) {
        margin += mulData[d];
      }

      if (Math.abs(margin) < EPS) {
        side[i] = random.nextBoolean();
        if (side[i]) {
          ++nLeft;
        } else {
          ++nRight;
        }
      } else if (margin > 0) {
        side[i] = false;
        ++nLeft;
      } else {
        side[i] = true;
        ++nRight;
      }
    }

    // Now that we have the counts allocate arrays
    int[] indicesLeft = new int[nLeft];
    int[] indicesRight = new int[nRight];

    // Populate the arrays with indices according to which side they fell on
    nLeft = 0;
    nRight = 0;
    for (int i = 0; i < side.length; ++i) {
      if (!side[i]) {
        indicesLeft[nLeft++] = indices[i];
      } else {
        indicesRight[nRight++] = indices[i];
      }
    }

    final Hyperplane hyperplane = new Hyperplane(hyperplaneInds, hyperplaneData);

    return new Object[]{indicesLeft, indicesRight, hyperplane, hyperplaneOffset};
  }


  static RandomProjectionTreeNode makeEuclideanTree(final Matrix data, final int[] indices, final Random random, final int leafSize) {
    if (indices.length > leafSize) {
      final Object[] erps = euclideanRandomProjectionSplit(data, indices, random);
      final int[] leftIndices = (int[]) erps[0];
      final int[] rightIndices = (int[]) erps[1];
      final Hyperplane hyperplane = new Hyperplane((float[]) erps[2]);
      final float offset = (float) erps[3];

      final RandomProjectionTreeNode leftNode = makeEuclideanTree(data, leftIndices, random, leafSize);
      final RandomProjectionTreeNode rightNode = makeEuclideanTree(data, rightIndices, random, leafSize);

      return new RandomProjectionTreeNode(null, false, hyperplane, offset, leftNode, rightNode);
    } else {
      return new RandomProjectionTreeNode(indices, true, null, null, null, null);
    }
  }

// def make_angular_tree(data, indices, random, leaf_size=30):
//     if indices.shape[0] > leaf_size:
//         left_indices, right_indices, hyperplane, offset = angular_random_projection_split(
//             data, indices, random
//         )

//         left_node = make_angular_tree(data, left_indices, random, leaf_size)
//         right_node = make_angular_tree(data, right_indices, random, leaf_size)

//         node = RandomProjectionTreeNode(
//             null, false, hyperplane, offset, left_node, right_node
//         )
//     else:
//         node = RandomProjectionTreeNode(indices, true, null, null, null, null)

//     return node


  static RandomProjectionTreeNode makeSparseEuclideanTree(final int[] inds, final int[] indptr, final float[] data, final int[] indices, final Random random, final int leafSize) {
    if (indices.length > leafSize) {
      final Object[] erps = sparseEuclideanRandomProjectionSplit(inds, indptr, data, indices, random);
      final int[] leftIndices = (int[]) erps[0];
      final int[] rightIndices = (int[]) erps[1];
      final Hyperplane hyperplane = (Hyperplane) erps[2];
      final float offset = (float) erps[3];

      RandomProjectionTreeNode leftNode = makeSparseEuclideanTree(inds, indptr, data, leftIndices, random, leafSize);
      RandomProjectionTreeNode rightNode = makeSparseEuclideanTree(inds, indptr, data, rightIndices, random, leafSize);

      return new RandomProjectionTreeNode(null, false, hyperplane, offset, leftNode, rightNode);
    } else {
      return new RandomProjectionTreeNode(indices, true, null, null, null, null);
    }
  }


// def make_sparse_angular_tree(inds, indptr, data, indices, random, leaf_size=30):
//     if indices.shape[0] > leaf_size:
//         left_indices, right_indices, hyperplane, offset = sparse_angular_random_projection_split(
//             inds, indptr, data, indices, random
//         )

//         left_node = make_sparse_angular_tree(
//             inds, indptr, data, left_indices, random, leaf_size
//         )
//         right_node = make_sparse_angular_tree(
//             inds, indptr, data, right_indices, random, leaf_size
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
//     random: array of int64, shape (3,)
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
  static RandomProjectionTreeNode makeTree(final Matrix data, final Random random, final int leaf_size, final boolean angular) {
    final boolean isSparse = data instanceof CsrMatrix;
    //final int indices = np.arange(data.shape[0]);
    final int[] indices = MathUtils.identity(data.rows());

    // Make a tree recursively until we get below the leaf size
    if (isSparse) {
      final CsrMatrix csrData = (CsrMatrix) data;
      final int[] inds = csrData.indices;
      final int[] indptr = csrData.indptr;
      final float[] spdata = csrData.data;

      if (angular) {
        throw new UnsupportedOperationException();
        //return make_sparse_angular_tree(inds, indptr, spdata, indices, random, leaf_size);
      } else {
        return makeSparseEuclideanTree(inds, indptr, spdata, indices, random, leaf_size);
      }
    } else {
      if (angular) {
        throw new UnsupportedOperationException();
        //return make_angular_tree(data, indices, random, leaf_size);
      } else {
        return makeEuclideanTree(data, indices, random, leaf_size);
      }
    }
  }


  // Determine the most number on non zeros in a hyperplane entry.
  static int maxSparseHyperplaneSize(RandomProjectionTreeNode tree) {
    if (tree.isLeaf()) {
      return 0;
    } else {
      return Math.max(
        tree.getHyperplane().mShape[1],
        Math.max(maxSparseHyperplaneSize(tree.getLeftChild()),
          maxSparseHyperplaneSize(tree.getRightChild()))
      );
    }
  }


  static int[] recursiveFlatten(final RandomProjectionTreeNode tree, final float[][][] hyperplanes, final float[] offsets, final int[][] children, final int[][] indices, int nodeNum, int leafNum) {
    if (tree.isLeaf()) {
      children[nodeNum][0] = -leafNum;
      //indices[leafNum, :tree.getIndices().shape[0]] =tree.getIndices();
      indices[leafNum] = tree.getIndices();
      leafNum += 1;
      return new int[]{nodeNum, leafNum};
    } else {
      if (tree.getHyperplane().mShape.length > 1) {
        // sparse case
        //hyperplanes[nodeNum][:, :tree.getHyperplane().shape[1]] =tree.getHyperplane();
        throw new UnsupportedOperationException();
      } else {
        hyperplanes[nodeNum] = null; //tree.getHyperplane().data; // todo
      }
      offsets[nodeNum] = tree.getOffset();
      children[nodeNum][0] = nodeNum + 1;
      final int old_node_num = nodeNum;
      final int[] t = recursiveFlatten(tree.getLeftChild(), hyperplanes, offsets, children, indices, nodeNum + 1, leafNum);
      nodeNum = t[0];
      leafNum = t[1];
      children[old_node_num][1] = nodeNum + 1;
      return recursiveFlatten(tree.getRightChild(), hyperplanes, offsets, children, indices, nodeNum + 1, leafNum);
    }
  }

  private static int[][] negOnes(final int a, final int b) {
    final int[][] res = new int[a][b];
    for (final int[] row : res) {
      Arrays.fill(row, -1);
    }
    return res;
  }


  static FlatTree flattenTree(final RandomProjectionTreeNode tree, final int leafSize) {
    final int nNodes = tree.numNodes();
    final int numLeaves = tree.numLeaves();

    final float[][][] hyperplanes;
    if (tree.getHyperplane().mShape.length > 1) {
      // sparse case
      final int maxHyperplaneNnz = maxSparseHyperplaneSize(tree);
      hyperplanes = new float[nNodes][tree.getHyperplane().mShape[0]][maxHyperplaneNnz];
    } else {
      hyperplanes = new float[nNodes][tree.getHyperplane().mShape[0]][1]; // todo ???
    }
    final float[] offsets = new float[nNodes];
    final int[][] children = negOnes(nNodes, 2);
    final int[][] indices = negOnes(numLeaves, leafSize);
    recursiveFlatten(tree, hyperplanes, offsets, children, indices, 0, 0);
    return new FlatTree(hyperplanes, offsets, children, indices);
  }


 static int selectSide(final float[] hyperplane, final float offset, final float[] point, final Random random) {
   float margin = offset;
   for (int d = 0; d < point.length; ++d) {
     margin += hyperplane[d] * point[d];
   }

   if (Math.abs(margin) < EPS) {
     return random.nextInt(2);
   } else if (margin > 0) {
     return 0;
   } else {
     return 1;
   }
 }

 static int[] searchFlatTree(final float[] point, final float[][] hyperplanes, final float[] offsets, final int[][] children, final int[][] indices, final Random random) {
   int node = 0;
   while (children[node][0] > 0) {
     final int side = selectSide(hyperplanes[node], offsets[node], point, random);
     if (side == 0) {
       node = children[node][0];
     } else {
       node = children[node][1];
     }
   }

   return indices[-children[node][0]];
 }


  //     """Build a random projection forest with ``nTrees``.
//
//     Parameters
//     ----------
//     data
//     n_neighbors
//     nTrees
//     random
//     angular
//
//     Returns
//     -------
//     forest: list
//         A list of random projection trees.
//     """
  static List<FlatTree> makeForest(final Matrix data, final int n_neighbors, final int nTrees, final Random random, final boolean angular) {
    final ArrayList<FlatTree> result = new ArrayList<>();
    final int leafSize = Math.max(10, n_neighbors);
    try {
      for (int i = 0; i < nTrees; ++i) {
        result.add(flattenTree(makeTree(data, random, leafSize, angular), leafSize));
      }
    } catch (RuntimeException e) {
      Utils.message("Random Projection forest initialisation failed due to recursion limit being reached. Something is a little strange with your data, and this may take longer than normal to compute.");
      throw e; // Python blindly continued from this point ... we die for now
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
//     random: array of int64, shape (3,)
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
