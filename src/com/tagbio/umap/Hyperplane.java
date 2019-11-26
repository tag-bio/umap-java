/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap;

/**
 * @author Sean A. Irvine
 */
class Hyperplane {

  final int[] inds;
  final float[] data;
  final int[] shape;

  Hyperplane(final int[] inds, final float[] data) {
    this.inds = inds;
    this.data = data;
    shape = new int[] {inds.length, 2};
  }

  Hyperplane(final float[] data) {
    this.inds = null;
    this.data = data;
    shape = new int[] {data.length};
  }
}
