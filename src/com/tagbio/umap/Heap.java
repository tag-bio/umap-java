package com.tagbio.umap;

/**
 * @author Sean A. Irvine
 */
class Heap {

  int[][] indices;
  float[][] weights;
  boolean[][] isNew;

  Heap(final int[][] indices, final float[][] weights) {
    this.indices = indices;
    this.weights = weights;
    isNew = new boolean[indices.length][indices[0].length];
  }

}
