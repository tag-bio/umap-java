/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap;

/**
 * Flattened tree.
 * @author Leland McInnes (Python)
 * @author Sean A. Irvine
 * @author Richard Littin
 */
class FlatTree {

  private final Object mHyperplanes;
  private final float[] mOffsets;
  private final int[][] mChildren;
  private final int[][] mIndices;

  FlatTree(final Object hyperplanes, final float[] offsets, final int[][] children, final int[][] indices) {
    mHyperplanes = hyperplanes;
    mOffsets = offsets;
    mChildren = children;
    mIndices = indices;
  }

  Object getHyperplanes() {
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
