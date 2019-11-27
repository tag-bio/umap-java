/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap;

import java.util.List;
import java.util.Random;

import com.tagbio.umap.metric.Metric;

/**
 * Nearest neighbor descent for a specified distance metric.
 * @author Leland McInnes (Python)
 * @author Sean A. Irvine
 * @author Richard Littin
 */
class NearestNeighborDescent {

  private final Metric mMetric;

  /**
   * Construct a nearest neighbor descent object for the given metric.
   * @param metric distance function
   */
  NearestNeighborDescent(final Metric metric) {
    this.mMetric = metric;
  }

  Heap descent(final Matrix data, final int nNeighbors, final Random random, final int maxCandidates, final boolean rp_tree_init, final int nIters, final int[][] leafArray, final boolean verbose) {
    return descent(data, nNeighbors, random, maxCandidates, rp_tree_init, nIters, leafArray, verbose, 0.001F, 0.5F);
  }

  Heap descent(final Matrix data, final int nNeighbors, final Random random, final int maxCandidates, final boolean rpTreeInit, final int nIters, final int[][] leafArray, final boolean verbose, final float delta, final float rho) {
    final int nVertices = data.rows();

    Heap currentGraph = Utils.makeHeap(data.rows(), nNeighbors);
    for (int i = 0; i < data.rows(); ++i) {
      final int[] indices = Utils.rejectionSample(nNeighbors, data.rows(), random);
      for (int j = 0; j < indices.length; ++j) {
        final float d = (float) mMetric.distance(data.row(i), data.row(indices[j]));
        Utils.heapPush(currentGraph, i, d, indices[j], true);
        Utils.heapPush(currentGraph, indices[j], d, i, true);
      }
    }
    if (rpTreeInit) {
      for (int n = 0; n < leafArray.length; ++n) {
        for (int i = 0; i < leafArray[n].length; ++i) {
          if (leafArray[n][i] < 0) {
            break;
          }
          for (int j = i + 1; j < leafArray[n].length; ++j) {
            if (leafArray[n][j] < 0) {
              break;
            }
            final float d = (float) mMetric.distance(data.row(leafArray[n][i]), data.row(leafArray[n][j]));
            Utils.heapPush(currentGraph, leafArray[n][i], d, leafArray[n][j], true);
            Utils.heapPush(currentGraph, leafArray[n][j], d, leafArray[n][i], true);
          }
        }
      }
    }

    for (int n = 0; n < nIters; ++n) {
      if (verbose) {
        Utils.message("NearestNeighborDescent: " + n + " / " + nIters);
      }

      final Heap candidateNeighbors = Utils.buildCandidates(currentGraph, nVertices, nNeighbors, maxCandidates, random);

      int c = 0;
      for (int i = 0; i < nVertices; ++i) {
        for (int j = 0; j < maxCandidates; ++j) {
          final int p = candidateNeighbors.indices[i][j];
          if (p < 0 || random.nextFloat() < rho) {
            continue;
          }
          for (int k = 0; k < maxCandidates; ++k) {
            final int q = candidateNeighbors.indices[i][k];
            if (q < 0 || !candidateNeighbors.isNew[i][j] && !candidateNeighbors.isNew[i][k]) {
              continue;
            }

            final float d = (float) mMetric.distance(data.row(p), data.row(q));
            c += Utils.heapPush(currentGraph, p, d, q, true);
            c += Utils.heapPush(currentGraph, q, d, p, true);
          }
        }
      }

      if (c <= delta * nNeighbors * data.rows()) {
        break;
      }
    }

    return Utils.deheapSort(currentGraph);
  }


  static Heap initialiseSearch(final List<FlatTree> forest, final Matrix data, final Matrix queryPoints, final int nNeighbors, final NearestNeighborRandomInit initFromRandom, NearestNeighborTreeInit initFromTree, final Random random) {
    final Heap results = Utils.makeHeap(queryPoints.rows(), nNeighbors);
    initFromRandom.init(nNeighbors, data, queryPoints, results, random);
    if (forest != null) {
      for (final FlatTree tree : forest) {
        initFromTree.init(tree, data, queryPoints, results, random);
      }
    }
    return results;
  }
}
