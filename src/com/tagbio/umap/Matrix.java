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
}
