package com.tagbio.umap;

import com.tagbio.umap.metric.Metric;

/**
 * @author Sean A. Irvine
 */
class NearestNeighborRandomInit {

  private final Metric dist;

  NearestNeighborRandomInit(final Metric dist) {
    this.dist = dist;
  }

  void init(final int n_neighbors, final Matrix data, final Matrix query_points, final Heap heap, final long[] rng_state) {
    for (int i = 0; i < query_points.rows(); ++i) {
      final int[] indices = Utils.rejection_sample(n_neighbors, data.shape[0], rng_state);
      for (int j = 0; j < indices.length; ++j) {
        if (indices[j] < 0) {
          continue;
        }
        final float d = (float) dist.distance(data.row(indices[j]), query_points.row(i));
        Utils.heap_push(heap, i, d, indices[j], true);
      }
    }
  }
}
