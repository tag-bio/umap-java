/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap;

import java.util.Random;

import com.tagbio.umap.metric.Metric;

/**
 * @author Sean A. Irvine
 */
class NearestNeighborRandomInit {

  private final Metric mDist;

  NearestNeighborRandomInit(final Metric dist) {
    mDist = dist;
  }

  void init(final int nNeighbors, final Matrix data, final Matrix queryPoints, final Heap heap, final Random rng_state) {
    for (int i = 0; i < queryPoints.rows(); ++i) {
      final int[] indices = Utils.rejectionSample(nNeighbors, data.shape[0], rng_state);
      for (int j = 0; j < indices.length; ++j) {
        if (indices[j] < 0) {
          continue;
        }
        final float d = (float) mDist.distance(data.row(indices[j]), queryPoints.row(i));
        Utils.heapPush(heap, i, d, indices[j], true);
      }
    }
  }
}
