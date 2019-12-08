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
class NearestNeighborTreeInit {

  private final Metric mDist;

  NearestNeighborTreeInit(final Metric dist) {
    mDist = dist;
  }

  void init(final FlatTree tree, final Matrix data, final Matrix queryPoints, final Heap heap, final Random random) {
    for (int i = 0; i < queryPoints.rows(); ++i) {
      final int[] indices = RandomProjectionTree.searchFlatTree(queryPoints.row(i), (float[][]) tree.getHyperplanes(), tree.getOffsets(), tree.getChildren(), tree.getIndices(), random);
      for (int j = 0; j < indices.length; ++j) {
        if (indices[j] < 0) {
          continue;
        }
        final float d = mDist.distance(data.row(indices[j]), queryPoints.row(i));
        heap.push(i, d, indices[j], true);
      }
    }
  }
}

