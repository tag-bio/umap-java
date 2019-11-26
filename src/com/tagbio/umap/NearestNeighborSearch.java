/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap;

import java.util.Set;
import java.util.TreeSet;

import com.tagbio.umap.metric.Metric;

/**
 * @author Sean A. Irvine
 */
class NearestNeighborSearch {

  private final Metric mDist;

  NearestNeighborSearch(final Metric dist) {
    mDist = dist;
  }

  Heap initialized_nnd_search(final Matrix data, final int[] indptr, final int[] indices, Heap initialization, final Matrix query_points) {

    for (int i = 0; i < query_points.rows(); ++i) {

      final Set<Integer> tried = new TreeSet<>();
      for (final int t : initialization.indices[i]) {
        tried.add(t);
      }

      while (true) {

        // Find smallest flagged vertex
        final int vertex = Utils.smallest_flagged(initialization, i);

        if (vertex == -1) {
          break;
        }
        final int[] candidates = new int[indptr[vertex + 1] - indptr[vertex]];
        System.arraycopy(indices, indptr[vertex], candidates, 0, candidates.length);
        for (int j = 0; j < candidates.length; ++j) {
          if (candidates[j] == vertex || candidates[j] == -1 || tried.contains(candidates[j])) {
            continue;
          }
          float d = (float) mDist.distance(data.row(candidates[j]), query_points.row(i));
          Utils.unchecked_heap_push(initialization, i, d, candidates[j], true);
          tried.add(candidates[j]);
        }
      }
    }

    return initialization;
  }
}
