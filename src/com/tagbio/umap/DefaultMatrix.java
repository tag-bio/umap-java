package com.tagbio.umap;

import java.util.Arrays;

/**
 * @author Sean A. Irvine
 */
class DefaultMatrix extends Matrix {

  private final float[][] data;

  DefaultMatrix(final float[][] vals) {
    super(vals.length, vals[0].length);
    data = vals;
  }

  DefaultMatrix(final int rows, final int cols) {
    super(rows, cols);
    data = new float[rows][cols];
  }

  DefaultMatrix(final int[] shape) {
    super(shape);
    data = new float[shape[0]][shape[1]];
  }

  @Override
  float get(final int row, final int col) {
    return data[row][col];
  }

  @Override
  void set(final int row, final int col, final float val) {
    data[row][col] = val;
  }

  @Override
  Matrix copy() {
    final float[][] copy = new float[data.length][];
    for (int k = 0; k < copy.length; ++k) {
      copy[k] = Arrays.copyOf(data[k], data[k].length);
    }
    return new DefaultMatrix(copy);
  }

  @Override
  float[] row(int row) {
    return Arrays.copyOf(data[row], data[row].length);
  }
}
