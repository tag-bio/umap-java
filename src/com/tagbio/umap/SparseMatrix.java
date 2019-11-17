package com.tagbio.umap;

/**
 * @author Sean A. Irvine
 */
class SparseMatrix {

  // todo perhaps should be an interface or at least abstract
  // todo also perhaps not always sparse?

  void eliminate_zeros() {
    // todo
  }

  SparseMatrix transpose() {
    // todo
    return null;
  }

  SparseMatrix multiply(final SparseMatrix m) {
    return null; // todo
  }

  SparseMatrix multiply(final double x) {
    // scalar multiply
    return null; // todo
  }

  SparseMatrix add(final SparseMatrix m) {
    // todo
    return null;
  }

  SparseMatrix subtract(final SparseMatrix m) {
    // todo
    return null;
  }

  long length() {
    // todo -- possibly total number of entries?
    return 0;
  }

  CooMatrix tocoo() {
    return null;
  }

  CsrMatrix tocsr() {
    return null;
  }

}
