/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap;

import java.util.Random;

import com.tagbio.umap.metric.Metric;

/**
 * @author Leland McInnes (Python)
 * @author Sean A. Irvine
 * @author Richard Littin
 */
class NearestNeighborRandomInit {

  private final Metric mDist;

  NearestNeighborRandomInit(final Metric dist) {
    mDist = dist;
  }

  void init(final int nNeighbors, final Matrix data, final Matrix queryPoints, final Heap heap, final Random random) {
    for (int i = 0; i < queryPoints.rows(); ++i) {
      final int[] indices = Utils.rejectionSample(nNeighbors, data.rows(), random);
      for (int j = 0; j < indices.length; ++j) {
        if (indices[j] < 0) { // todo is this check necessary?
          continue;
        }
        final float d = mDist.distance(data.row(indices[j]), queryPoints.row(i));
        heap.push(i, d, indices[j], true);
      }
    }
  }
}
