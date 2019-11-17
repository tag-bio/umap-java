package com.tagbio.umap;

/**
 * @author Sean A. Irvine
 */
class CsrMatrix extends SparseMatrix {
  // todo -- replacement fo scipy csr_matrix

  CsrMatrix(final float[] vals, final int[] rows, final int[] cols, final int[] lengths) {
  }

  // todo I think this is internal rep of data for this form of matrix -- currently some direct external access
  int[] indptr;
  int[] indices;
  float[] data;
}
