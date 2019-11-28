/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap;

/**
 * @author Sean A. Irvine
 * @author Richard Littin
 */
class Normalize {

  // After sklearn.preprocessing.normalize
  // todo only need support "max" and "l1" for now
  // todo handle sparse input, esp. Csr
  // todo method of normalization could be enum
  // todo ditch this after additing to matrix

  static Matrix normalize(final Matrix data, final String method) {
    if (!"max".equals(method) && !"l1".equals(method)) {
      throw new UnsupportedOperationException();
    }
    if ("max".equals(method)) {
      return data.rowNormalize();
    }
    // todo -- this is row-wise normalization
    // for each row find maximum element, then normalize row by maximum (resp. l1-norm)
    // return COPY of data (i.e. do not modify in place)
    // if input is CsrMatrix, then output should be CsrMatrix
    return null;
  }
}
