package com.tagbio.umap;

/**
 * @author Sean A. Irvine
 */
class CsrMatrix extends Matrix {
  // todo -- replacement fo scipy csr_matrix

  // todo I think this is internal rep of data for this form of matrix -- currently some direct external access
  int[] indptr;
  int[] indices;
  float[] data;

  CsrMatrix(final float[] vals, final int[] rows, final int[] cols, final int[] lengths) {
    super(lengths);
  }

  boolean has_sorted_indices() {
    // todo
    return true;
  }

  void sort_indices() {
    // todo
  }

  @Override
  float get(final int row, final int col) {
    throw new UnsupportedOperationException();
  }

  @Override
  void set(final int row, final int col, final float val) {
    throw new UnsupportedOperationException();
  }
}
