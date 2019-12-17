/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.util.Arrays;

/**
 * Node in a random projection tree.
 * @author Sean A. Irvine
 * @author Richard Littin
 */
class RandomProjectionTreeNode {

  private final int[] mIndices;
  private final Hyperplane mHyperplane;
  private final Float mOffset;
  private final RandomProjectionTreeNode mLeftChild;
  private final RandomProjectionTreeNode mRightChild;

  RandomProjectionTreeNode(final int[] indices, final Hyperplane hyperplane, final Float offset, final RandomProjectionTreeNode leftChild, final RandomProjectionTreeNode rightChild) {
    mIndices = indices;
    mHyperplane = hyperplane;
    mOffset = offset;
    mLeftChild = leftChild;
    mRightChild = rightChild;
  }

  Hyperplane getHyperplane() {
    return mHyperplane;
  }

  boolean isLeaf() {
    return mLeftChild == null && mRightChild == null;
  }

  int numNodes() {
    return 1 + (mLeftChild != null ? mLeftChild.numNodes() : 0) + (mRightChild != null ? mRightChild.numNodes() : 0);
  }

  int numLeaves() {
    return isLeaf() ? 1 : mLeftChild.numLeaves() + mRightChild.numLeaves();
  }

  RandomProjectionTreeNode getLeftChild() {
    return mLeftChild;
  }

  RandomProjectionTreeNode getRightChild() {
    return mRightChild;
  }

  int[] getIndices() {
    return mIndices;
  }

  Float getOffset() {
    return mOffset;
  }

  private static int[][] negOnes(final int a, final int b) {
    final int[][] res = new int[a][b];
    for (final int[] row : res) {
      Arrays.fill(row, -1);
    }
    return res;
  }

  private int[] recursiveFlatten(final Object hyperplanes, final float[] offsets, final int[][] children, final int[][] indices, final int nodeNum, final int leafNum) {
    if (isLeaf()) {
      children[nodeNum][0] = -leafNum;
      //indices[leafNum, :tree.getIndices().shape[0]] =tree.getIndices();
      indices[leafNum] = getIndices();
      return new int[]{nodeNum, leafNum + 1};
    } else {
      if (getHyperplane().shape().length > 1) {
        // sparse case
        ((float[][][]) hyperplanes)[nodeNum] = new float[][] {getHyperplane().data()}; // todo dubious
        //hyperplanes[nodeNum][:, :tree.getHyperplane().shape[1]] =tree.getHyperplane();
      } else {
        ((float[][]) hyperplanes)[nodeNum] = getHyperplane().data();
      }
      offsets[nodeNum] = getOffset();
      children[nodeNum][0] = nodeNum + 1;
      final int[] flattenInfo = getLeftChild().recursiveFlatten(hyperplanes, offsets, children, indices, nodeNum + 1, leafNum);
      children[nodeNum][1] = flattenInfo[0] + 1;
      return getRightChild().recursiveFlatten(hyperplanes, offsets, children, indices, flattenInfo[0] + 1, flattenInfo[1]);
    }
  }

  // Determine the most number on non zeros in a hyperplane entry.
  private int maxSparseHyperplaneSize() {
    if (isLeaf()) {
      return 0;
    } else {
      return Math.max(getHyperplane().shape()[1], Math.max(getLeftChild().maxSparseHyperplaneSize(), getRightChild().maxSparseHyperplaneSize()));
    }
  }

  FlatTree flatten() {
    final int nNodes = numNodes();
    final int numLeaves = numLeaves();

    final Object hyperplanes;
    if (getHyperplane().shape().length > 1) {
      // sparse case
      final int maxHyperplaneNnz = maxSparseHyperplaneSize();
      hyperplanes = new float[nNodes][getHyperplane().shape()[0]][maxHyperplaneNnz];
    } else {
      hyperplanes = new float[nNodes][getHyperplane().shape()[0]];
    }
    final float[] offsets = new float[nNodes];
    final int[][] children = negOnes(nNodes, 2);
    final int[][] indices = new int[numLeaves][];
    recursiveFlatten(hyperplanes, offsets, children, indices, 0, 0);
    return new FlatTree(hyperplanes, offsets, children, indices);
  }

}
