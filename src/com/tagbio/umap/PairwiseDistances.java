/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap;

import com.tagbio.umap.metric.Metric;
import com.tagbio.umap.metric.PrecomputedMetric;

/**
 * @author Sean A. Irvine
 */
class PairwiseDistances {

  // replacement for sklearn.pairwise_distances
  // todo testing
  // todo possibly smarter version for sparse data
  // todo can this assume symmetric metric (i.e. d(x,y)=d(y,x)) ?

  private PairwiseDistances() { }

  static Matrix pairwise_distances(final Matrix x, final Metric metric) {
    // todo special metric precomputed
    if (PrecomputedMetric.SINGLETON.equals(metric)) {
      return x;
    }
    // todo keywords
    // todo potential parallel
    //Utils.message("Starting distance calculation");
    final int n = x.shape()[0];
    final float[][] distances = new float[n][n];
    for (int k = 0; k < n; ++k) {
      final float[] xk = x.row(k);
      for (int j = 0; j < n; ++j) {
        distances[k][j] = (float) metric.distance(xk, x.row(j));
      }
    }
    //Utils.message("Finished distance calculation");
    return new DefaultMatrix(distances);
  }

  static Matrix pairwise_distances(final Matrix x, final Matrix y, final Metric metric) {
    if (PrecomputedMetric.SINGLETON.equals(metric)) {
      throw new IllegalArgumentException("Cannot use this method with precomputed");
    }
    // todo keywords
    // todo potential parallel
    final int xn = x.shape()[0];
    final int yn = y.shape()[0];
    final float[][] distances = new float[xn][yn];
    for (int k = 0; k < xn; ++k) {
      final float[] xk = x.row(k);
      for (int j = 0; j < yn; ++j) {
        distances[k][j] = (float) metric.distance(xk, y.row(j));
      }
    }
    return new DefaultMatrix(distances);
  }

}
