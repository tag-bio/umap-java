package com.tagbio.umap;

/**
 * @author Sean A. Irvine
 */
class RandomProjectionTreeNode {

  private final int[] mIndices;
  private final boolean mIsLeaf;
  private final Hyperplane mHyperplane;
  private final Float mOffset;
  private final RandomProjectionTreeNode mLeftChild;
  private final RandomProjectionTreeNode mRightChild;

  RandomProjectionTreeNode(final int[] indices, final boolean isLeaf, final Hyperplane hyperplane, final Float offset, final RandomProjectionTreeNode leftChild, final RandomProjectionTreeNode rightChild) {
    mIndices = indices;
    mIsLeaf = isLeaf;
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

  int num_nodes() {
    return 1 + (mLeftChild != null ? mLeftChild.num_nodes() : 0) + (mRightChild != null ? mRightChild.num_nodes() : 0);
  }

  int num_leaves() {
    return isLeaf() ? 1 : mLeftChild.num_leaves() + mRightChild.num_leaves();
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
