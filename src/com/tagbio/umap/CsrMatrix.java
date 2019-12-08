/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap;

import java.util.Arrays;

/**
 * A form of sparse matrix where only non-zero entries are explicitly recorded.
 * This format is compatible with the Python scipy <code>csr_matrix</code> format.
 * @author Sean A. Irvine
 * @author Richard Littin
 */
class CsrMatrix extends Matrix {

  private int[] mIndptr;  // indptr[row] to indptr[row + 1] locations of cols in indices
  private int[] mIndices; // positions of actual data
  private float[] mData;

  CsrMatrix(final float[] data, final int[] indptr, final int[] indices, final int rowCount, final int colCount) {
    super(rowCount, colCount);
    mIndptr = indptr;
    mIndices = indices;
    mData = data;
  }

  int[] indptr() {
    return Arrays.copyOf(mIndptr, mIndptr.length);
  }

  int[] indicies() {
    return Arrays.copyOf(mIndices, mIndices.length);
  }

  float[] data() {
    return Arrays.copyOf(mData, mData.length);
  }

  @Override
  float get(final int row, final int col) {
    final int colStart = mIndptr[row];
    final int colEnd = mIndptr[row + 1];
    for (int p = colStart; p < colEnd; ++p) {
      if (mIndices[p] == col) {
        return mData[p];
      }
    }
    return 0;
  }

  @Override
  void set(final int row, final int col, final float val) {
    throw new UnsupportedOperationException();
  }

  @Override
  Matrix copy() {
    return new CsrMatrix(Arrays.copyOf(mData, mData.length), Arrays.copyOf(mIndptr, mIndptr.length), Arrays.copyOf(mIndices, mIndices.length), rows(), cols());
  }

  @Override
  CsrMatrix toCsr() {
    return this;
  }

  @Override
  Matrix add(final Matrix m) {
    // todo this could do this without using super
    return super.add(m).toCsr();
  }

  @Override
  Matrix subtract(final Matrix m) {
    // todo this could do this without using super
    return super.subtract(m).toCsr();
  }

  @Override
  Matrix multiply(final Matrix m) {
    // todo this could do this without using super
    return super.multiply(m).toCsr();
  }

  @Override
  Matrix transpose() {
    // todo this could do this without using super
    return super.transpose().toCsr();
  }

  @Override
  Matrix multiply(final float x) {
    final float[] newData = Arrays.copyOf(mData, mData.length);
    for (int i = 0; i < newData.length; ++i) {
      newData[i] *= x;
    }
    return new CsrMatrix(newData, mIndptr, mIndices, rows(), cols());
  }

  @Override
  Matrix rowNormalize() {
    final float[] d = new float[mData.length];
    for (int row = 0; row < rows(); ++row) {
      float max = mData[mIndptr[row]];
      for (int j = mIndptr[row] + 1; j < mIndptr[row + 1]; ++j) {
        max = Math.max(max, mData[j]);
      }
      for (int j = mIndptr[row]; j < mIndptr[row + 1]; ++j) {
        d[j] = mData[j] / max;
      }
    }
    // Note would be safer to cop mIndptr and mIndices arrays
    return new CsrMatrix(d, mIndptr, mIndices, rows(), cols());
  }

  boolean hasSortedIndices() {
    // todo
    throw new UnsupportedOperationException();
  }

  void sortIndices() {
    // todo
    throw new UnsupportedOperationException();
  }

}
