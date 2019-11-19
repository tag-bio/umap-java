package com.tagbio.umap;

/**
 * @author Sean A. Irvine
 */
class Matrix {

  // todo perhaps should be an interface or at least abstract
  // subclasses CooMatrix, CsrMatrix (pos LilMatrix) + a non-sparse float[][] backed implementation

  float get(final int row, final int col) {
    // todo
    return 0;
  }

  void set(final int row, final int col, final float val) {
    // todo need to be supported for all types
  }

  void eliminate_zeros() {
    // todo
  }

  Matrix transpose() {
    // todo
    return null;
  }

  Matrix multiply(final Matrix m) {
    return null; // todo
  }

  Matrix multiply(final double x) {
    // scalar multiply
    return null; // todo
  }

  Matrix add(final Matrix m) {
    // todo
    return null;
  }

  Matrix subtract(final Matrix m) {
    // todo
    return null;
  }

  long length() {
    // todo --  total number of entries
    return 0;
  }

  CooMatrix tocoo() {
    return null;
  }

  CsrMatrix tocsr() {
    return null;
  }

  int[] shape() {
    // todo length of dimensions
    return null;
  }

  Matrix copy() {
    // todo this should be a copy of the matrix of same type -- generics on params?
    return this;
  }

  float[] row(int row) {
    // return entire row as a vector
    return null;
  }

  Matrix take(int[] indicies) {
    // todo return elements from array along selected axes
    return null;
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
    return null;
  }

  Matrix dot(Matrix x) {
    // todo dot product
    return null;
  }

  Matrix inv() {
    // todo invert
    return null;
  }

  Matrix triu(int k) {
    // todo Upper triangle of an array.
    // Return a copy of a matrix with the elements below the `k`-th diagonal zeroed.
    // may want tri and tril versions as well
    return null;
  }

  Float max() {
    // todo maximum value in matrix
    // this might only be wanted for float[], not matrix...
    // min as well?
    return null;
  }

  Float sum() {
    // todo sum of values in matrix
    // return double?
    return null;
  }

  Matrix max(final Matrix other) {
    // todo element-wise maximum between two matrices
    // todo other should be same shape as this
    return null;
  }
}
