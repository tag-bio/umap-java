package com.tagbio.umap;

/**
 * @author Sean A. Irvine
 */
class CsrMatrix extends Matrix {
  // todo -- replacement fo scipy csr_matrix

  CsrMatrix(final float[] vals, final int[] rows, final int[] cols, final int[] lengths) {
  }

  // todo I think this is internal rep of data for this form of matrix -- currently some direct external access
  int[] indptr;
  int[] indices;
  float[] data;

  boolean has_sorted_indices() {
    // todo
    return true;
  }

  void sort_indices() {
    // todo
  }
}
