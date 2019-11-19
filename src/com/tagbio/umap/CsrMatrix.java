package com.tagbio.umap;

/**
 * @author Sean A. Irvine
 */
class CsrMatrix extends Matrix {
  // todo -- replacement fo scipy csr_matrix

  // todo I think this is internal rep of data for this form of matrix -- currently some direct external access
  int[] indptr;  // indptr[row] to indptr[row + 1] locations of cols in indices
  int[] indices; // positions of actual data
  float[] data;

  CsrMatrix(final int[] lengths) {
    super(lengths);
  }

  boolean has_sorted_indices() {
    // todo
    throw new UnsupportedOperationException();
  }

  void sort_indices() {
    // todo
    throw new UnsupportedOperationException();
  }

  @Override
  float get(final int row, final int col) {
    final int colStart = indptr[row];
    final int colEnd = indptr[row + 1];
    for (int p = colStart; p <= colEnd; ++p) {
      if (indices[p] == col) {
        return data[p];
      }
    }
    return 0;
  }

  @Override
  void set(final int row, final int col, final float val) {
    throw new UnsupportedOperationException();
  }
}
