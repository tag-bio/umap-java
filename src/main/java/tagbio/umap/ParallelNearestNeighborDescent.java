/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import tagbio.umap.metric.Metric;

/**
 * Nearest neighbor descent for a specified distance metric.
 * Nondeterministic parallel version.
 * @author Leland McInnes (Python)
 * @author Sean A. Irvine
 * @author Richard Littin
 */
class ParallelNearestNeighborDescent extends  NearestNeighborDescent {

  private final int mThreads;

  /**
   * Construct a nearest neighbor descent object for the given metric.
   * @param metric distance function
   * @param threads number of threads
   */
  ParallelNearestNeighborDescent(final Metric metric, final int threads) {
    super(metric);
    if (threads < 1) {
      throw new IllegalArgumentException();
    }
    mThreads = threads;
  }

  private void waitForThreads(final Thread... threads) {
    try {
      for (final Thread thread : threads) {
        thread.join();
      }
    } catch (InterruptedException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  Heap descent(final Matrix data, final int nNeighbors, final Random random, final int maxCandidates, final boolean rpTreeInit, final int nIters, final int[][] leafArray) {
    return descent(data, nNeighbors, random, maxCandidates, rpTreeInit, nIters, leafArray, 0.001F, 0.5F);
  }

  @Override
  Heap descent(final Matrix data, final int nNeighbors, final Random random, final int maxCandidates, final boolean rpTreeInit, final int nIters, final int[][] leafArray, final float delta, final float rho) {
    UmapProgress.incTotal(nIters);

    final Thread[] threads = new Thread[mThreads];
    final int nVertices = data.rows();
    final int chunkSize = (nVertices + mThreads - 1) / mThreads;
    final Heap currentGraph = new Heap(data.rows(), nNeighbors);
    for (int t = 0; t < mThreads; ++t) {
      final int lo = t * chunkSize;
      final int hi = Math.min((t + 1) * chunkSize, nVertices);
      threads[t] = new Thread(() -> {
        for (int i = lo; i < hi; ++i) {
          final float[] iRow = data.row(i);
          for (final int index : Utils.rejectionSample(nNeighbors, data.rows(), random)) {
            final float d = mMetric.distance(iRow, data.row(index));
            currentGraph.push(i, d, index, true);
            currentGraph.push(index, d, i, true);
          }
        }
      });
      threads[t].start();
    }
    waitForThreads(threads);

    if (rpTreeInit) {
      final int cs = (leafArray.length + mThreads - 1) / mThreads;
      for (int t = 0; t < mThreads; ++t) {
        final int lo = t * cs;
        final int hi = Math.min((t + 1) * cs, leafArray.length);
        threads[t] = new Thread(() -> {
          //System.out.println("T: " + lo + ":" + hi + " : " + leafArray.length);
          for (int l = lo; l < hi; ++l) {
            int[] leaf = leafArray[l];
            for (int i = 0; i < leaf.length; ++i) {
              final float[] iRow = data.row(leaf[i]);
              for (int j = i + 1; j < leaf.length; ++j) {
                final float d = mMetric.distance(iRow, data.row(leaf[j]));
                currentGraph.push(leaf[i], d, leaf[j], true);
                currentGraph.push(leaf[j], d, leaf[i], true);
              }
            }
          }
        });
        threads[t].start();
      }
      waitForThreads(threads);
    }

    final boolean[] rejectStatus = new boolean[maxCandidates];
    for (int n = 0; n < nIters; ++n) {
      if (mVerbose) {
        Utils.message("NearestNeighborDescent: " + n + " / " + nIters);
      }

      final Heap candidateNeighbors = currentGraph.buildCandidates(nVertices, nNeighbors, maxCandidates, random);

      final AtomicInteger totalC = new AtomicInteger();
      for (int t = 0; t < mThreads; ++t) {
        final int lo = t * chunkSize;
        final int hi = Math.min((t + 1) * chunkSize, nVertices);
        threads[t] = new Thread(() -> {
          int c = 0;
          for (int i = lo; i < hi; ++i) {
            for (int j = 0; j < maxCandidates; ++j) {
              rejectStatus[j] = random.nextFloat() < rho;
            }

            for (int j = 0; j < maxCandidates; ++j) {
              final int p = candidateNeighbors.index(i, j);
              if (p < 0) {
                continue;
              }
              for (int k = 0; k <= j; ++k) {
                final int q = candidateNeighbors.index(i, k);
                if (q < 0) {
                  continue;
                }
                if (rejectStatus[j] && rejectStatus[k]) {
                  continue;
                }
                if (!candidateNeighbors.isNew(i, j) && !candidateNeighbors.isNew(i, k)) {
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
            totalC.addAndGet(c);
          }
        });
        threads[t].start();
      }
      waitForThreads(threads);

      final int c = totalC.get();

//      // todo delete this debug
//      final int[][] idx = currentGraph.indices();
//      final float[][] w = currentGraph.weights();
//      float sum = 0;
//      for (int i = 0; i < idx.length; ++i) {
//        for (int j = 0; j < idx[i].length; ++j) {
//          if (idx[i][j] >= 0) {
//            sum += w[i][j];
//          }
//        }
//      }
//      System.out.println(n + " heap-weight " + sum + " c=" + c);

      if (c <= delta * nNeighbors * data.rows()) {
        UmapProgress.update(nIters - n);
        break;
      }
      UmapProgress.update();
    }

    return currentGraph.deheapSort();
  }
}
