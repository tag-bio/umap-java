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
    if (rows.length != cols.length || rows.length != vals.length) {
      throw new IllegalArgumentException();
    }
    row = rows;
    col = cols;
    data = vals;
  }

  void sum_duplicates() {
    // todo -- semantics?  possibly just merge identical entries?
    throw new UnsupportedOperationException();
  }

  @Override
  float get(final int r, final int c) {
    // todo this could be made faster if can assume sorted row[] col[]
    // todo there may be duplicate coords, need to sum result?
    for (int k = 0; k < row.length; ++k) {
      if (row[k] == r && col[k] == c) {
        return data[k];
      }
    }
    return 0;
  }

  @Override
  void set(final int row, final int col, final float val) {
    throw new UnsupportedOperationException();
  }

  @Override
  Matrix transpose() {
    // todo note this is not copying the arrays -- might be a mutability issue
    return new CooMatrix(data, col, row, new int[] {shape[1], shape[0]});
  }

  @Override
  CooMatrix tocoo() {
    return this;
  }
}
