/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

/**
 * @author Sean A. Irvine
 */
class RandomProjectionTreeNode {

  private final int[] mIndices;
  //private final boolean mIsLeaf;
  private final Hyperplane mHyperplane;
  private final Float mOffset;
  private final RandomProjectionTreeNode mLeftChild;
  private final RandomProjectionTreeNode mRightChild;

  RandomProjectionTreeNode(final int[] indices, final boolean isLeaf, final Hyperplane hyperplane, final Float offset, final RandomProjectionTreeNode leftChild, final RandomProjectionTreeNode rightChild) {
    mIndices = indices;
    //mIsLeaf = isLeaf;
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
}
