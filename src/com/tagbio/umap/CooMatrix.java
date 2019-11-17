package com.tagbio.umap;

/**
 * @author Sean A. Irvine
 */
class CooMatrix extends SparseMatrix {
  // todo -- replacement fo scipy coo_matrix

  CooMatrix(final float[] vals, final int[] rows, final int[] cols, final int[] lengths) {
  }

  // todo I think this is internal rep of data for this form of matrix -- currently some direct external access
  int[] row;
  int[] col;
  float[] data;

  void sum_duplicates() {
    // todo -- semantics?  possibly just merge identical entries?
  }
}
