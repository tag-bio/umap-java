package com.tagbio.umap;

/**
 * Base class for matrices.
 * @author Sean A. Irvine
 */
abstract class Matrix {

  // todo perhaps should be an interface or at least abstract
  // subclasses CooMatrix, CsrMatrix (pos LilMatrix) + a non-sparse float[][] backed implementation

  protected int[] shape;

  Matrix(final int... shape) {
    this.shape = shape;
  }

  abstract float get(final int row, final int col);

  abstract void set(final int row, final int col, final float val);

  int[] shape() {
    return shape;
  }

  long length() {
    long len = 0;
    for (final int dim : shape()) {
      len *= dim;
    }
    return len;
  }

  void eliminate_zeros() {
    throw new UnsupportedOperationException();
  }

  Matrix transpose() {
    // todo
    throw new UnsupportedOperationException();
  }

  Matrix multiply(final Matrix m) {
    throw new UnsupportedOperationException();
  }

  Matrix multiply(final double x) {
    // scalar multiply
    throw new UnsupportedOperationException();
  }

  Matrix add(final Matrix m) {
    // todo
    throw new UnsupportedOperationException();
  }

  Matrix subtract(final Matrix m) {
    // todo
    throw new UnsupportedOperationException();
  }

  CooMatrix tocoo() {
    throw new UnsupportedOperationException();
  }

  CsrMatrix tocsr() {
    throw new UnsupportedOperationException();
  }

  Matrix copy() {
    // todo this should be a copy of the matrix of same type -- generics on params?
    throw new UnsupportedOperationException();
  }

  float[] row(int row) {
    // return entire row as a vector
    throw new UnsupportedOperationException();
  }

  Matrix take(int[] indicies) {
    // todo return elements from array along selected axes
    throw new UnsupportedOperationException();
  }

  static Matrix eye(int n, int m, int k) {
    // todo
    /*
    Return a 2-D array with ones on the diagonal and zeros elsewhere.
    N : int
      Number of rows in the output.
    M : int, optional
      Number of columns in the output. If None, defaults to `N`.
    k : int, optional
      Index of the diagonal: 0 (the default) refers to the main diagonal,
      a positive value refers to an upper diagonal, and a negative value
      to a lower diagonal.
    */
    throw new UnsupportedOperationException();
  }

  Matrix dot(Matrix x) {
    // todo dot product
    throw new UnsupportedOperationException();
  }

  Matrix inv() {
    // todo invert
    throw new UnsupportedOperationException();
  }

  Matrix triu(int k) {
    // todo Upper triangle of an array.
    // Return a copy of a matrix with the elements below the `k`-th diagonal zeroed.
    // may want tri and tril versions as well
    throw new UnsupportedOperationException();
  }

  Float max() {
    // todo maximum value in matrix
    // this might only be wanted for float[], not matrix...
    // min as well?
    throw new UnsupportedOperationException();
  }

  Float sum() {
    // todo sum of values in matrix
    // return double?
    throw new UnsupportedOperationException();
  }

  Matrix max(final Matrix other) {
    // todo element-wise maximum between two matrices
    // todo other should be same shape as this
    throw new UnsupportedOperationException();
  }
}
