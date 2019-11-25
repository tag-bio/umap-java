package com.tagbio.umap;

import com.tagbio.umap.metric.Metric;

/**
 * @author Sean A. Irvine
 */
class NearestNeighborTreeInit {

  private final Metric dist;

  NearestNeighborTreeInit(final Metric dist) {
    this.dist = dist;
  }

  void init(final FlatTree tree, final Matrix data, final Matrix query_points, final Heap heap, final long[] rng_state) {
    for (int i = 0; i < query_points.rows(); ++i) {
      final int[] indices = RpTree.search_flat_tree(query_points.row(i), tree.getHyperplanes()[0], tree.getOffsets(), tree.getChildren(), tree.getIndices(), rng_state); // todo !!! xxx hyperplanes[0]
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

