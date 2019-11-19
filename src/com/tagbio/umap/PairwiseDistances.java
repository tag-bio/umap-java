package com.tagbio.umap;

import java.util.Map;

import com.tagbio.umap.metric.Metric;

/**
 * @author Sean A. Irvine
 */
class PairwiseDistances {

  // replacement for sklearn.pairwise_distances
  // todo testing
  // todo possibly smarter version for sparse data

  private PairwiseDistances() { }

  private static float[] instance(final Matrix data, final int row) {
    final float[] r = new float[data.shape()[1]];
    for (int col = 0; col < r.length; ++col) {
      r[col] = data.get(row, col);
    }
    return r;
  }

  static Matrix pairwise_distances(final Matrix x, final Metric metric, final Map<String, Object> keywords) {
    // todo special metric precomputed
    if (metric.equals("precomputed")) {
      return x;
    }
    // todo keywords
    // todo potential parallel
    final int n = x.shape()[0];
    final float[][] distances = new float[n][n];
    for (int k = 0; k < n; ++k) {
      final float[] xk = instance(x, k);
      for (int j = 0; j < n; ++j) {
        distances[k][j] = (float) metric.distance(xk, instance(x, j));
      }
    }
    return new DefaultMatrix(distances);
  }

  static Matrix pairwise_distances(final Matrix x, final Matrix y, final Metric metric, final Map<String, Object> keywords) {
    // todo special metric precomputed
    if (metric.equals("precomputed")) {
      throw new IllegalArgumentException("Cannot use this method with precomputed");
    }
    // todo keywords
    // todo potential parallel
    final int xn = x.shape()[0];
    final int yn = y.shape()[0];
    final float[][] distances = new float[xn][yn];
    for (int k = 0; k < xn; ++k) {
      final float[] xk = instance(x, k);
      for (int j = 0; j < yn; ++j) {
        distances[k][j] = (float) metric.distance(xk, instance(y, j));
      }
    }
    return new DefaultMatrix(distances);
  }

}
