/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.util.Set;
import java.util.TreeSet;

import tagbio.umap.metric.Metric;

/**
 * @author Leland McInnes (Python)
 * @author Sean A. Irvine
 * @author Richard Littin
 */
class NearestNeighborSearch {

  private final Metric mDist;

  NearestNeighborSearch(final Metric dist) {
    mDist = dist;
  }

  Heap initializedNndSearch(final Matrix data, final SearchGraph searchGraph, Heap initialization, final Matrix queryPoints) {
    for (int i = 0; i < queryPoints.rows(); ++i) {

      final Set<Integer> tried = new TreeSet<>();
      for (final int t : initialization.indices()[i]) {
        tried.add(t);
      }

      while (true) {

        // Find smallest flagged vertex
        final int vertex = initialization.smallestFlagged(i);

        if (vertex == -1) {
          break;
        }
        //final int[] candidates = new int[indptr[vertex + 1] - indptr[vertex]];
        //System.arraycopy(indices, indptr[vertex], candidates, 0, candidates.length);
        for (int candidate : searchGraph.row(vertex)) {
          if (candidate == vertex || candidate == -1 || tried.contains(candidate)) { // todo is this -1 needed
            continue;
          }
          final float d = mDist.distance(data.row(candidate), queryPoints.row(i));
          initialization.uncheckedHeapPush(i, d, candidate, true);
          tried.add(candidate);
        }
      }
    }

    return initialization;
  }
}
