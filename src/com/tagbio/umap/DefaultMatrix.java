package com.tagbio.umap;

/**
 * @author Sean A. Irvine
 */
class DefaultMatrix extends Matrix {

  private final float[][] data;

  DefaultMatrix(final float[][] vals) {
    super(vals.length, vals[0].length);
    data = vals;
  }

  @Override
  float get(final int row, final int col) {
    return data[row][col];
  }

  @Override
  void set(final int row, final int col, final float val) {
    data[row][col] = val;
  }
}
