package com.tagbio.umap;

import java.util.Arrays;

/**
 * @author Sean A. Irvine
 */
class CsrMatrix extends Matrix {
  // todo -- replacement fo scipy csr_matrix

  // todo I think this is internal rep of data for this form of matrix -- currently some direct external access
  int[] indptr;  // indptr[row] to indptr[row + 1] locations of cols in indices
  int[] indices; // positions of actual data
  float[] data;

  CsrMatrix(final float[] data, final int[] indptr, final int[] indices, final int[] lengths) {
    super(lengths);
    this.indptr = indptr;
    this.indices = indices;
    this.data = data;
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
    for (int p = colStart; p < colEnd; ++p) {
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

  @Override
  Matrix copy() {
    return new CsrMatrix(Arrays.copyOf(data, data.length), Arrays.copyOf(indptr, indptr.length), Arrays.copyOf(indices, indices.length), Arrays.copyOf(shape, shape.length));
  }

  @Override
  CsrMatrix tocsr() {
    return this;
  }

  @Override
  Matrix add(final Matrix m) {
    // todo this could do this without using super
    return super.add(m).tocsr();
  }

  @Override
  Matrix subtract(final Matrix m) {
    // todo this could do this without using super
    return super.subtract(m).tocsr();
  }

  @Override
  Matrix multiply(final Matrix m) {
    // todo this could do this without using super
    return super.multiply(m).tocsr();
  }

  @Override
  Matrix transpose() {
    // todo this could do this without using super
    return super.transpose().tocsr();
  }

  @Override
  Matrix multiply(final float x) {
    final float[] newData = Arrays.copyOf(data, data.length);
    for (int i = 0; i < newData.length; ++i) {
      newData[i] *= x;
    }
    return new CsrMatrix(newData, indptr, indices, shape);
  }
}
