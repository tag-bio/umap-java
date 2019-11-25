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
