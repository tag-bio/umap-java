/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.util.Random;

import tagbio.umap.metric.Metric;

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
      for (int index : indices) {
        if (index < 0) { // todo is this check necessary?
          continue;
        }
        final float d = mDist.distance(data.row(index), queryPoints.row(i));
        heap.push(i, d, index, true);
      }
    }
  }
}
