/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap;

/**
 * Container for a hyperplane.
 * @author Sean A. Irvine
 */
class Hyperplane {

  final int[] mInds;
  final float[] mData;
  final int[] mShape;

  Hyperplane(final int[] inds, final float[] data) {
    mInds = inds;
    mData = data;
    mShape = new int[] {inds.length, 2};
  }

  Hyperplane(final float[] data) {
    mInds = null;
    mData = data;
    mShape = new int[] {data.length};
  }
}
