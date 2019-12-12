/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * Random projection trees.
 * @author Leland McInnes (Python)
 * @author Sean A. Irvine
 * @author Richard Littin
 */
final class RandomProjectionTree {

  private RandomProjectionTree() { }

  // Used for a floating point "nearly zero" comparison
  private static final float EPS = 1e-8F;

  /**
   * Given a set of <code>indices</code> for data points from <code>data</code>, create
   * a random hyperplane to split the data, returning two arrays indices
   * that fall on either side of the hyperplane. This is the basis for a
   * random projection tree, which simply uses this splitting recursively.
   * This particular split uses cosine distance to determine the hyperplane
   * and which side each data sample falls on.
   * @param data array of shape <code>(nSamples, nFeatures)</code>. The original data to be split
   * @param indices array of shape <code>(treeNodeSize)</code>
   * The indices of the elements in the <code>data</code> array that are to
   * be split in the current operation.
   * @param random randomness source
   * @return The elements of <code>indices</code> that fall on the "left" side of the
   * random hyperplane.
   */
  private static Object[] angularRandomProjectionSplit(final Matrix data, final int[] indices, final Random random) {
    final int dim = data.cols();

    // Select two random points, set the hyperplane between them
    final int leftIndex = random.nextInt(indices.length);
    int rightIndex = random.nextInt(indices.length);
    if (leftIndex == rightIndex) {
      rightIndex = (rightIndex + 1) % indices.length;
    }
    final int left = indices[leftIndex];
    final int right = indices[rightIndex];

    float leftNorm = Utils.norm(data.row(left));
    float rightNorm = Utils.norm(data.row(right));

    if (Math.abs(leftNorm) < EPS) {
      leftNorm = 1;
    }

    if (Math.abs(rightNorm) < EPS) {
      rightNorm = 1;
    }

    // Compute the normal vector to the hyperplane (the vector between the two points)
    final float[] hyperplaneVector = new float[dim];

    for (int d = 0; d < dim; ++d) {
      hyperplaneVector[d] = (data.get(left, d) / leftNorm) - (data.get(right, d) / rightNorm);
    }

    float hyperplaneNorm = Utils.norm(hyperplaneVector);
    if (Math.abs(hyperplaneNorm) < EPS) {
      hyperplaneNorm = 1;
    }

    for (int d = 0; d < dim; ++d) {
      hyperplaneVector[d] /= hyperplaneNorm;
    }

    // For each point compute the margin (project into normal vector)
    // If we are on lower side of the hyperplane put in one pile, otherwise
    // put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    int nLeft = 0;
    int nRight = 0;
    final boolean[] side = new boolean[indices.length];
    for (int i = 0; i < indices.length; ++i) {
      float margin = 0;
      for (int d = 0; d < dim; ++d) {
        margin += hyperplaneVector[d] * data.get(indices[i], d);
      }

      if (Math.abs(margin) < EPS) {
        side[i] = random.nextBoolean();
        if (side[i]) {
          nRight += 1;
        } else {
          nLeft += 1;
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
      if (side[i]) {
        indicesRight[nRight++] = indices[i];
      } else {
        indicesLeft[nLeft++] = indices[i];
      }
    }

    return new Object[]{indicesLeft, indicesRight, hyperplaneVector, null};
  }


  /**
   * Given a set of <code>indices</code> for data points from <code>data</code>, create
   * a random hyperplane to split the data, returning two arrays indices
   * that fall on either side of the hyperplane. This is the basis for a
   * random projection tree, which simply uses this splitting recursively.
   * This particular split uses Euclidean distance to determine the hyperplane
   * and which side each data sample falls on.
   * @param data array of shape <code>(nSamples, nFeatures)</code>. The original data to be split
   * @param indices array of shape <code>(treeNodeSize)</code>
   * The indices of the elements in the <code>data</code> array that are to
   * be split in the current operation.
   * @param random randomness source
   * @return The elements of <code>indices</code> that fall on the "left" side of the
   * random hyperplane.
   */
  private static Object[] euclideanRandomProjectionSplit(final Matrix data, final int[] indices, final Random random) {
    final int dim = data.cols();

    // Select two random points, set the hyperplane between them
    final int leftIndex = random.nextInt(indices.length);
    int rightIndex = random.nextInt(indices.length);
    if (leftIndex == rightIndex) {
      rightIndex = (rightIndex + 1) % indices.length;
    }
    final int left = indices[leftIndex];
    final int right = indices[rightIndex];

    // Compute the normal vector to the hyperplane (the vector between
    // the two points) and the offset from the origin
    float hyperplaneOffset = 0;
    final float[] hyperplaneVector = new float[dim];

    for (int d = 0; d < dim; ++d) {
      hyperplaneVector[d] = data.get(left, d) - data.get(right, d);
      hyperplaneOffset -= hyperplaneVector[d] * (data.get(left, d) + data.get(right, d)) / 2.0;
    }

    // For each point compute the margin (project into normal vector, add offset)
    // If we are on lower side of the hyperplane put in one pile, otherwise
    // put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    int nLeft = 0;
    int nRight = 0;
    final boolean[] side = new boolean[indices.length];
    for (int i = 0; i < indices.length; ++i) {
      float margin = hyperplaneOffset;
      for (int d = 0; d < dim; ++d) {
        margin += hyperplaneVector[d] * data.get(indices[i], d);
      }
      if (Math.abs(margin) < EPS) {
        side[i] = random.nextBoolean();
        if (side[i]) {
          ++nRight;
        } else {
          ++nLeft;
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
      if (side[i]) {
        indicesRight[nRight++] = indices[i];
      } else {
        indicesLeft[nLeft++] = indices[i];
      }
    }
    return new Object[]{indicesLeft, indicesRight, hyperplaneVector, hyperplaneOffset};
  }

  /**
   * Given a set of <code>indices</code> for data points from <code>data</code>, create
   * a random hyperplane to split the data, returning two arrays indices
   * that fall on either side of the hyperplane. This is the basis for a
   * random projection tree, which simply uses this splitting recursively.
   * This particular split uses cosine distance to determine the hyperplane
   * and which side each data sample falls on.
   * @param inds CSR format index array of the matrix,
   * @param indptr CSR format index pointer array of the matrix.
   * @param data CSR format data array of the matrix
   * @param random randomness source
   * @return The elements of <code>indices</code> that fall on the "left" side of the
   * random hyperplane.
   */
  private static Object[] sparseAngularRandomProjectionSplit(final int[] inds, final int[] indptr, final float[] data, final int[] indices, final Random random) {
    // Select two random points, set the hyperplane between them
    final int leftIndex = random.nextInt(indices.length);
    int rightIndex = random.nextInt(indices.length);
    if (leftIndex == rightIndex) {
      rightIndex = (rightIndex + 1) % indices.length;
    }
    final int left = indices[leftIndex];
    final int right = indices[rightIndex];

    final int[] leftInds = MathUtils.subarray(inds, indptr[left], indptr[left + 1]);
    final float[] leftData = MathUtils.subarray(data, indptr[left], indptr[left + 1]);
    final int[] rightInds = MathUtils.subarray(inds, indptr[right], indptr[right + 1]);
    final float[] rightData = MathUtils.subarray(data, indptr[right], indptr[right + 1]);

    float leftNorm = Utils.norm(leftData);
    float rightNorm = Utils.norm(rightData);

    if (Math.abs(leftNorm) < EPS) {
      leftNorm = 1;
    }

    if (Math.abs(rightNorm) < EPS) {
      rightNorm = 1;
    }

    // Compute the normal vector to the hyperplane (the vector between the two points)
    final float[] normalizedLeftData = MathUtils.divide(leftData, leftNorm);
    final float[] normalizedRightData = MathUtils.divide(rightData, rightNorm);
    final Object[] sd = Sparse.sparseDiff(leftInds, normalizedLeftData, rightInds, normalizedRightData);
    final int[] hyperplaneInds = (int[]) sd[0];
    final float[] hyperplaneData = (float[]) sd[1];

    float hyperplaneNorm = Utils.norm(hyperplaneData);
    if (Math.abs(hyperplaneNorm) < EPS) {
      hyperplaneNorm = 1;
    }
    for (int d = 0; d < hyperplaneData.length; ++d) {
      hyperplaneData[d] /= hyperplaneNorm;
    }

    // For each point compute the margin (project into normal vector)
    // If we are on lower side of the hyperplane put in one pile, otherwise
    // put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    int nLeft = 0;
    int nRight = 0;
    final boolean[] side = new boolean[indices.length];
    for (int i = 0; i < indices.length; ++i) {
      float margin = 0;

      final int[] iInds = MathUtils.subarray(inds, indptr[indices[i]], indptr[indices[i] + 1]);
      final float[] iData = MathUtils.subarray(data, indptr[indices[i]], indptr[indices[i] + 1]);

      final Object[] spm = Sparse.multiply(hyperplaneInds, hyperplaneData, iInds, iData);
      //final int[] mulInds = (int[]) spm[0];
      final float[] mulData = (float[]) spm[1];
      for (final float d : mulData) {
        margin += d;
      }
      if (Math.abs(margin) < EPS) {
        side[i] = random.nextBoolean();
        if (side[i]) {
          ++nRight;
        } else {
          ++nLeft;
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
      if (side[i]) {
        indicesRight[nRight++] = indices[i];
      } else {
        indicesLeft[nLeft++] = indices[i];
      }
    }

    final Hyperplane hyperplane = new Hyperplane(hyperplaneInds, hyperplaneData);

    return new Object[]{indicesLeft, indicesRight, hyperplane, null};
  }

  /**
   * Given a set of <code>indices</code> for data points from <code>data</code>, create
   * a random hyperplane to split the data, returning two arrays indices
   * that fall on either side of the hyperplane. This is the basis for a
   * random projection tree, which simply uses this splitting recursively.
   * This particular split uses Euclidean distance to determine the hyperplane
   * and which side each data sample falls on.
   * @param inds CSR format index array of the matrix,
   * @param indptr CSR format index pointer array of the matrix.
   * @param data CSR format data array of the matrix
   * @param random randomness source
   * @return The elements of <code>indices</code> that fall on the "left" side of the
   * random hyperplane.
   */
  private static Object[] sparseEuclideanRandomProjectionSplit(final int[] inds, final int[] indptr, final float[] data, final int[] indices, final Random random) {
    // Select two random points, set the hyperplane between them
    final int leftIndex = random.nextInt(indices.length);
    int rightIndex = random.nextInt(indices.length);
    if (leftIndex == rightIndex) {
      rightIndex = (rightIndex + 1) % indices.length;
    }
    final int left = indices[leftIndex];
    final int right = indices[rightIndex];

    final int[] leftInds = MathUtils.subarray(inds, indptr[left], indptr[left + 1]);
    final float[] leftData = MathUtils.subarray(data, indptr[left], indptr[left + 1]);
    final int[] rightInds = MathUtils.subarray(inds, indptr[right], indptr[right + 1]);
    final float[] rightData = MathUtils.subarray(data, indptr[right], indptr[right + 1]);

    // Compute the normal vector to the hyperplane (the vector between
    // the two points) and the offset from the origin
    float hyperplaneOffset = 0;
    final Object[] sd = Sparse.sparseDiff(leftInds, leftData, rightInds, rightData);
    final int[] hyperplaneInds = (int[]) sd[0];
    final float[] hyperplaneData = (float[]) sd[1];
    final Object[] ss = Sparse.sparseSum(leftInds, leftData, rightInds, rightData);
    final Object[] sm = Sparse.multiply(hyperplaneInds, hyperplaneData, (int[]) ss[0], MathUtils.divide((float[]) ss[1], 2.0F));
    //final int[] offsetInds = (int[]) sm[0];
    final float[] offsetData = (float[]) sm[1];

    for (final float d : offsetData) {
      hyperplaneOffset -= d;
    }

    // For each point compute the margin (project into normal vector, add offset)
    // If we are on lower side of the hyperplane put in one pile, otherwise
    // put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    int nLeft = 0;
    int nRight = 0;
    final boolean[] side = new boolean[indices.length];
    for (int i = 0; i < indices.length; ++i) {
      float margin = hyperplaneOffset;
      final int[] iInds = MathUtils.subarray(inds, indptr[indices[i]], indptr[indices[i] + 1]);
      final float[] iData = MathUtils.subarray(data, indptr[indices[i]], indptr[indices[i] + 1]);

      final Object[] spm = Sparse.multiply(hyperplaneInds, hyperplaneData, iInds, iData);
      //final int[] mulInds = (int[]) spm[0];
      final float[] mulData = (float[]) spm[1];
      for (final float d : mulData) {
        margin += d;
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
    final int[] indicesLeft = new int[nLeft];
    final int[] indicesRight = new int[nRight];

    // Populate the arrays with indices according to which side they fell on
    nLeft = 0;
    nRight = 0;
    for (int i = 0; i < side.length; ++i) {
      if (side[i]) {
        indicesRight[nRight++] = indices[i];
      } else {
        indicesLeft[nLeft++] = indices[i];
      }
    }

    final Hyperplane hyperplane = new Hyperplane(hyperplaneInds, hyperplaneData);

    return new Object[]{indicesLeft, indicesRight, hyperplane, hyperplaneOffset};
  }

  private static RandomProjectionTreeNode makeEuclideanTree(final Matrix data, final int[] indices, final Random random, final int leafSize) {
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

  private static RandomProjectionTreeNode makeAngularTree(final Matrix data, final int[] indices, final Random random, final int leafSize) {
    if (indices.length > leafSize) {
      final Object[] erps = angularRandomProjectionSplit(data, indices, random);
      final int[] leftIndices = (int[]) erps[0];
      final int[] rightIndices = (int[]) erps[1];
      final Hyperplane hyperplane = new Hyperplane((float[]) erps[2]);
      final float offset = (float) erps[3];

      final RandomProjectionTreeNode leftNode = makeAngularTree(data, leftIndices, random, leafSize);
      final RandomProjectionTreeNode rightNode = makeAngularTree(data, rightIndices, random, leafSize);

      return new RandomProjectionTreeNode(null, false, hyperplane, offset, leftNode, rightNode);
    } else {
      return new RandomProjectionTreeNode(indices, true, null, null, null, null);
    }
  }

  private static RandomProjectionTreeNode makeSparseEuclideanTree(final int[] inds, final int[] indptr, final float[] data, final int[] indices, final Random random, final int leafSize) {
    if (indices.length > leafSize) {
      final Object[] erps = sparseEuclideanRandomProjectionSplit(inds, indptr, data, indices, random);
      final int[] leftIndices = (int[]) erps[0];
      final int[] rightIndices = (int[]) erps[1];
      final Hyperplane hyperplane = (Hyperplane) erps[2];
      final float offset = (float) erps[3];

      final RandomProjectionTreeNode leftNode = makeSparseEuclideanTree(inds, indptr, data, leftIndices, random, leafSize);
      final RandomProjectionTreeNode rightNode = makeSparseEuclideanTree(inds, indptr, data, rightIndices, random, leafSize);

      return new RandomProjectionTreeNode(null, false, hyperplane, offset, leftNode, rightNode);
    } else {
      return new RandomProjectionTreeNode(indices, true, null, null, null, null);
    }
  }

  private static RandomProjectionTreeNode makeSparseAngularTree(final int[] inds, final int[] indptr, final float[] data, final int[] indices, final Random random, final int leafSize) {
    if (indices.length > leafSize) {
      final Object[] erps = sparseAngularRandomProjectionSplit(inds, indptr, data, indices, random);
      final int[] leftIndices = (int[]) erps[0];
      final int[] rightIndices = (int[]) erps[1];
      final Hyperplane hyperplane = (Hyperplane) erps[2];
      final float offset = (float) erps[3];

      final RandomProjectionTreeNode leftNode = makeSparseAngularTree(inds, indptr, data, leftIndices, random, leafSize);
      final RandomProjectionTreeNode rightNode = makeSparseAngularTree(inds, indptr, data, rightIndices, random, leafSize);

      return new RandomProjectionTreeNode(null, false, hyperplane, offset, leftNode, rightNode);
    } else {
      return new RandomProjectionTreeNode(indices, true, null, null, null, null);
    }
  }

  /**
   * Construct a random projection tree based on <code>data</code> with leaves
   * of size at most <code>leafSize</code>.
   * @param data array of shape <code>(nSamples, nFeatures)</code>
   * The original data to be split
   * @param random randomness source
   * @param leafSize The maximum size of any leaf node in the tree. Any node in the tree
   * with more than <code>leafSize</code> will be split further to create child
   * nodes.
   * @param angular Whether to use cosine/angular distance to create splits in the tree,
   * or Euclidean distance
   * @return A random projection tree node which links to its child nodes. This
   * provides the full tree below the returned node.
   */
  private static RandomProjectionTreeNode makeTree(final Matrix data, final Random random, final int leafSize, final boolean angular) {
    final boolean isSparse = data instanceof CsrMatrix;
    final int[] indices = MathUtils.identity(data.rows());

    // Make a tree recursively until we get below the leaf size
    if (isSparse) {
      final CsrMatrix csrData = (CsrMatrix) data;
      final int[] inds = csrData.indicies();
      final int[] indptr = csrData.indptr();
      final float[] spdata = csrData.data();

      if (angular) {
        return makeSparseAngularTree(inds, indptr, spdata, indices, random, leafSize);
      } else {
        return makeSparseEuclideanTree(inds, indptr, spdata, indices, random, leafSize);
      }
    } else {
      if (angular) {
        return makeAngularTree(data, indices, random, leafSize);
      } else {
        return makeEuclideanTree(data, indices, random, leafSize);
      }
    }
  }


  // Determine the most number on non zeros in a hyperplane entry.
  private static int maxSparseHyperplaneSize(RandomProjectionTreeNode tree) {
    if (tree.isLeaf()) {
      return 0;
    } else {
      return Math.max(tree.getHyperplane().shape()[1], Math.max(maxSparseHyperplaneSize(tree.getLeftChild()), maxSparseHyperplaneSize(tree.getRightChild())));
    }
  }


  private static int[] recursiveFlatten(final RandomProjectionTreeNode tree, final Object hyperplanes, final float[] offsets, final int[][] children, final int[][] indices, final int nodeNum, final int leafNum) {
    if (tree.isLeaf()) {
      children[nodeNum][0] = -leafNum;
      //indices[leafNum, :tree.getIndices().shape[0]] =tree.getIndices();
      indices[leafNum] = tree.getIndices();
      return new int[]{nodeNum, leafNum + 1};
    } else {
      if (tree.getHyperplane().shape().length > 1) {
        // sparse case
        ((float[][][]) hyperplanes)[nodeNum] = new float[][] {tree.getHyperplane().data()}; // todo dubious
        //hyperplanes[nodeNum][:, :tree.getHyperplane().shape[1]] =tree.getHyperplane();
        throw new UnsupportedOperationException();
      } else {
        ((float[][]) hyperplanes)[nodeNum] = tree.getHyperplane().data();
      }
      offsets[nodeNum] = tree.getOffset();
      children[nodeNum][0] = nodeNum + 1;
      final int[] flattenInfo = recursiveFlatten(tree.getLeftChild(), hyperplanes, offsets, children, indices, nodeNum + 1, leafNum);
      children[nodeNum][1] = flattenInfo[0] + 1;
      return recursiveFlatten(tree.getRightChild(), hyperplanes, offsets, children, indices, flattenInfo[0] + 1, flattenInfo[1]);
    }
  }

  private static int[][] negOnes(final int a, final int b) {
    final int[][] res = new int[a][b];
    for (final int[] row : res) {
      Arrays.fill(row, -1);
    }
    return res;
  }

  private static FlatTree flattenTree(final RandomProjectionTreeNode tree, final int leafSize) {
    final int nNodes = tree.numNodes();
    final int numLeaves = tree.numLeaves();

    final Object hyperplanes;
    if (tree.getHyperplane().shape().length > 1) {
      // sparse case
      final int maxHyperplaneNnz = maxSparseHyperplaneSize(tree);
      hyperplanes = new float[nNodes][tree.getHyperplane().shape()[0]][maxHyperplaneNnz];
    } else {
      hyperplanes = new float[nNodes][tree.getHyperplane().shape()[0]];
    }
    final float[] offsets = new float[nNodes];
    final int[][] children = negOnes(nNodes, 2);
    final int[][] indices = new int[numLeaves][];
    recursiveFlatten(tree, hyperplanes, offsets, children, indices, 0, 0);
    return new FlatTree(hyperplanes, offsets, children, indices);
  }

 private static boolean selectSide(final float[] hyperplane, final float offset, final float[] point, final Random random) {
   float margin = offset;
   for (int d = 0; d < point.length; ++d) {
     margin += hyperplane[d] * point[d];
   }

   if (Math.abs(margin) < EPS) {
     return random.nextBoolean();
   } else {
     return margin <= 0;
   }
 }

 static int[] searchFlatTree(final float[] point, final float[][] hyperplanes, final float[] offsets, final int[][] children, final int[][] indices, final Random random) {
   int node = 0;
   while (children[node][0] > 0) {
     final boolean side = selectSide(hyperplanes[node], offsets[node], point, random);
     if (side) {
       node = children[node][1];
     } else {
       node = children[node][0];
     }
   }
   return indices[-children[node][0]];
 }

  /**
   * Build a random projection forest with specified number of trees.
   * @param data instances
   * @param nNeighbors number of nearest neighbours
   * @param nTrees number of trees
   * @param random randomness source
   * @param angular true for cosine metric, otherwise Euclidean
   * @return list of random projection trees
   */
  static List<FlatTree> makeForest(final Matrix data, final int nNeighbors, final int nTrees, final Random random, final boolean angular) {
    final Random[] randoms = Utils.splitRandom(random, nTrees);  // insure same set of random numbers for 1 and multiple threads

    final ArrayList<FlatTree> result = new ArrayList<>();
    final int leafSize = Math.max(10, nNeighbors);
    try {
      for (int i = 0; i < nTrees; ++i) {
        result.add(flattenTree(makeTree(data, randoms[i], leafSize, angular), leafSize));
        UmapProgress.update();
      }
    } catch (RuntimeException e) {
      Utils.message("Random Projection forest initialisation failed due to recursion limit being reached. Something is a little strange with your data, and this may take longer than normal to compute.");
      throw e; // Python blindly continued from this point ... we die for now
    }
    return result;
  }

  static List<FlatTree> makeForest(final Matrix data, final int nNeighbors, final int nTrees, final Random random, final boolean angular, int threads) {
    if (threads == 1) {
      return makeForest(data, nNeighbors, nTrees, random, angular);
    }
    final Random[] randoms = Utils.splitRandom(random, nTrees);  // insure same set of random numbers for 1 and multiple threads

    Thread.UncaughtExceptionHandler h = new Thread.UncaughtExceptionHandler() {
      @Override
      public void uncaughtException(Thread th, Throwable ex) {
        Utils.message("XXXXRandom Projection forest initialisation failed due to recursion limit being reached. Something is a little strange with your data, and this may take longer than normal to compute.");
        throw new RuntimeException(ex); // Python blindly continued from this point ... we die for now
      }
    };

    final ExecutorService executor = Executors.newFixedThreadPool(threads);
    final List<Future<FlatTree>> futures = new ArrayList<>();

    final int leafSize = Math.max(10, nNeighbors);
    for (final Random rand : randoms) {  // randoms.length == nTrees
      futures.add(executor.submit(() -> flattenTree(makeTree(data, rand, leafSize, angular), leafSize)));
    }

    final ArrayList<FlatTree> result = new ArrayList<>();
    try {
      for (Future<FlatTree> future : futures) {
        result.add(future.get());
        UmapProgress.update();
      }
    } catch (InterruptedException | ExecutionException ex) {
      Utils.message("Random Projection forest initialisation failed due to recursion limit being reached. Something is a little strange with your data, and this may take longer than normal to compute.");
      throw new RuntimeException(ex); // Python blindly continued from this point ... we die for now
    }
    return result;
  }


  /**
   * Generate an array of sets of candidate nearest neighbors by
   * constructing a random projection forest and taking the leaves of all the
   * trees. Any given tree has leaves that are a set of potential nearest
   * neighbors. Given enough trees the set of all such leaves gives a good
   * likelihood of getting a good set of nearest neighbors in composite. Since
   * such a random projection forest is inexpensive to compute, this can be a
   * useful means of seeding other nearest neighbor algorithms.
   * @param rpForest forest
   * @return array of shape <code>(nLeaves, max(10, nNeighbors))</code>
   * Each row of leaf array is a list of indices found in a given leaf.
   */
  static int[][] rptreeLeafArray(final List<FlatTree> rpForest) {
    if (rpForest.size() > 0) {
      final List<int[]> leafArray = new ArrayList<>();
      for (final FlatTree flatTree : rpForest) {
        Collections.addAll(leafArray, flatTree.getIndices());
      }
      return leafArray.toArray(new int[0][]);
    } else {
      return new int[0][];
    }
  }
}
