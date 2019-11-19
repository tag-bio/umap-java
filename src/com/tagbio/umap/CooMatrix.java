package com.tagbio.umap;

/**
 * @author Sean A. Irvine
 */
class CooMatrix extends Matrix {
  // todo -- replacement fo scipy coo_matrix

  // todo I think this is internal rep of data for this form of matrix -- currently some direct external access
  int[] row;
  int[] col;
  float[] data;

  CooMatrix(final float[] vals, final int[] rows, final int[] cols, final int[] lengths) {
    super(lengths);
  }

  void sum_duplicates() {
    // todo -- semantics?  possibly just merge identical entries?
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
