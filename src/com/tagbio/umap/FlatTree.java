package com.tagbio.umap;

/**
 * @author Sean A. Irvine
 */
class FlatTree {

  private final float[][][] mHyperplanes;
  private final float[] mOffsets;
  private final int[][] mChildren;
  private final int[][] mIndices;

  FlatTree(final float[][][] hyperplanes, final float[] offsets, final int[][] children, final int[][] indices) {
    mHyperplanes = hyperplanes;
    mOffsets = offsets;
    mChildren = children;
    mIndices = indices;
  }

  float[][][] getHyperplanes() {
    return mHyperplanes;
  }

  float[] getOffsets() {
    return mOffsets;
  }

  int[][] getChildren() {
    return mChildren;
  }

  int[][] getIndices() {
    return mIndices;
  }
}
