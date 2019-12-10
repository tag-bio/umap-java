/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.util.List;
import java.util.Random;

import tagbio.umap.metric.Metric;

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
    mMetric = metric;
  }

  Heap descent(final Matrix data, final int nNeighbors, final Random random, final int maxCandidates, final boolean rpTreeInit, final int nIters, final int[][] leafArray, final boolean verbose) {
    return descent(data, nNeighbors, random, maxCandidates, rpTreeInit, nIters, leafArray, verbose, 0.001F, 0.5F);
  }

  Heap descent(final Matrix data, final int nNeighbors, final Random random, final int maxCandidates, final boolean rpTreeInit, final int nIters, final int[][] leafArray, final boolean verbose, final float delta, final float rho) {
    UmapProgress.incTotal(nIters);

    final int nVertices = data.rows();
    final Heap currentGraph = new Heap(data.rows(), nNeighbors);
    // todo parallel -- note use of random, care needed to maintain determinism -- i.e. how to split generator, sync on .push in heap
    for (int i = 0; i < data.rows(); ++i) {
      final float[] iRow = data.row(i);
      for (final int index : Utils.rejectionSample(nNeighbors, data.rows(), random)) {
        final float d = mMetric.distance(iRow, data.row(index));
        currentGraph.push(i, d, index, true);
        currentGraph.push(index, d, i, true);
      }
    }

    if (rpTreeInit) {
      // todo parallel
      for (final int[] leaf : leafArray) {
        for (int i = 0; i < leaf.length; ++i) {
          final float[] iRow = data.row(leaf[i]);
          for (int j = i + 1; j < leaf.length; ++j) {
            final float d = mMetric.distance(iRow, data.row(leaf[j]));
            currentGraph.push(leaf[i], d, leaf[j], true);
            currentGraph.push(leaf[j], d, leaf[i], true);
          }
        }
      }
    }

    for (int n = 0; n < nIters; ++n) {
      if (verbose) {
        Utils.message("NearestNeighborDescent: " + n + " / " + nIters);
      }

      final Heap candidateNeighbors = currentGraph.buildCandidates(nVertices, nNeighbors, maxCandidates, random);

      int c = 0;
      for (int i = 0; i < nVertices; ++i) {
        for (int j = 0; j < maxCandidates; ++j) {
          final int p = candidateNeighbors.index(i, j);
          if (p < 0 || random.nextFloat() < rho) {
            continue;
          }
          for (int k = 0; k < maxCandidates; ++k) {
            final int q = candidateNeighbors.index(i, k);
            if (q < 0 || (!candidateNeighbors.isNew(i, j) && !candidateNeighbors.isNew(i, k))) {
              continue;
            }

            final float d = mMetric.distance(data.row(p), data.row(q));
            if (currentGraph.push(p, d, q, true)) {
              ++c;
            }
            if (currentGraph.push(q, d, p, true)) {
              ++c;
            }
          }
        }
      }

      if (c <= delta * nNeighbors * data.rows()) {
        UmapProgress.update(nIters - n);
        break;
      }
      UmapProgress.update();
    }

    return currentGraph.deheapSort();
  }


  static Heap initialiseSearch(final List<FlatTree> forest, final Matrix data, final Matrix queryPoints, final int nNeighbors, final NearestNeighborRandomInit initFromRandom, NearestNeighborTreeInit initFromTree, final Random random) {
    final Heap results = new Heap(queryPoints.rows(), nNeighbors);
    initFromRandom.init(nNeighbors, data, queryPoints, results, random);
    if (forest != null) {
      for (final FlatTree tree : forest) {
        initFromTree.init(tree, data, queryPoints, results, random);
      }
    }
    return results;
  }
}
