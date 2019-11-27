/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap;

import java.util.Arrays;
import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.Random;

/**
 * Utility functions.
 * @author Leland McInnes (Python)
 * @author Sean A. Irvine
 * @author Richard Littin
 */
class Utils {

  private Utils() {
  }

  /**
   * Get the current date and time as a string of the form
   * <code>YYYY-MM-DD hh:mm:ss</code>.
   * @return date string
   */
  private static String now() {
    final StringBuilder sb = new StringBuilder();
    final Calendar cal = new GregorianCalendar();
    sb.append(cal.get(Calendar.YEAR)).append('-');
    final int month = 1 + cal.get(Calendar.MONTH);
    if (month < 10) {
      sb.append('0');
    }
    sb.append(month).append('-');
    final int date = cal.get(Calendar.DATE);
    if (date < 10) {
      sb.append('0');
    }
    sb.append(date).append(' ');
    final int hour = cal.get(Calendar.HOUR_OF_DAY);
    if (hour < 10) {
      sb.append('0');
    }
    sb.append(hour).append(':');
    final int min = cal.get(Calendar.MINUTE);
    if (min < 10) {
      sb.append('0');
    }
    sb.append(min).append(':');
    final int sec = cal.get(Calendar.SECOND);
    if (sec < 10) {
      sb.append('0');
    }
    sb.append(sec).append(' ');
    return sb.toString();
  }

  /**
   * Print a dated message on standard output.
   * @param message message to print
   */
  static void message(final String message) {
    System.out.println(now() + message);
  }

  /**
   * A fast computation of knn indices.
   * @param instances array of shape (nSamples, nFeatures)
   * @param nNeighbors the number of nearest neighbors to compute for each sample in <code>instances</code>
   * @return array of shape (nSamples, nNeighbors) containing the indices of the <code>nNeighbours</code>
   * closest points in the dataset.
   */
  static int[][] fastKnnIndices(final Matrix instances, final int nNeighbors) {
    final int[][] knnIndices = new int[instances.rows()][nNeighbors];
    for (int row = 0; row < instances.rows(); ++row) {
      final int[] v = MathUtils.argsort(instances.row(row));
      knnIndices[row] = Arrays.copyOf(v, nNeighbors);
    }
    return knnIndices;
  }


// @numba.njit()
// def norm(vec):
//     """Compute the (standard l2) norm of a vector.

//     Parameters
//     ----------
//     vec: array of shape (dim,)

//     Returns
//     -------
//     The l2 norm of vec.
//     """
//     result = 0.0
//     for i in range(vec.shape[0]):
//         result += vec[i] ** 2
//     return np.sqrt(result)


//     """Generate n_samples many integers from 0 to pool_size such that no
//     integer is selected twice. The duplication constraint is achieved via
//     rejection sampling.

//     Parameters
//     ----------
//     n_samples: int
//         The number of random samples to select from the pool

//     pool_size: int
//         The size of the total pool of candidates to sample from

//     rng_state: array of int64, shape (3,)
//         Internal state of the random number generator

  //     Returns
//     -------
//     sample: array of shape(nSamples,)
//         The ``nSamples`` randomly selected elements from the pool.
  static int[] rejectionSample(final int nSamples, final int poolSize, final Random random) {
    final int[] result = new int[nSamples];
    for (int i = 0; i < nSamples; ++i) {
      boolean rejectSample = true;
      int j = -1;
      while (rejectSample) {
        j = random.nextInt(poolSize);
        boolean ok = true;
        for (int k = 0; k < i; ++k) {
          if (j == result[k]) {
            ok = false;
            break;
          }
        }
        if (ok) {
          rejectSample = false;
        }
      }
      result[i] = j;
    }
    return result;
  }

  private static void fill(final float[][] a, final float val) {
    for (final float[] b : a) {
      Arrays.fill(b, val);
    }
  }

  private static void fill(final int[][] a, final int val) {
    for (final int[] b : a) {
      Arrays.fill(b, val);
    }
  }

//     Constructor for the numba enabled heap objects. The heaps are used
//     for approximate nearest neighbor search, maintaining a list of potential
//     neighbors sorted by their distance. We also flag if potential neighbors
//     are newly added to the list or not. Internally this is stored as
//     a single ndarray; the first axis determines whether we are looking at the
//     array of candidate indices, the array of distances, or the flag array for
//     whether elements are new or not. Each of these arrays are of shape
//     (``n_points``, ``size``)

//     Parameters
//     ----------
//     n_points: int
//         The number of data points to track in the heap.

//     size: int
//         The number of items to keep on the heap for each data point.

  //     Returns
//     -------
//     heap: An ndarray suitable for passing to other numba enabled heap functions.
  static Heap makeHeap(int nPoints, int size) {
    final Heap result = new Heap(new int[nPoints][size], new float[nPoints][size]);
    fill(result.indices, -1);
    fill(result.weights, Float.POSITIVE_INFINITY);
    //fill(result[2], 0);
    return result;
  }


//     """Push a new element onto the heap. The heap stores potential neighbors
//     for each data point. The ``row`` parameter determines which data point we
//     are addressing, the ``weight`` determines the distance (for heap sorting),
//     the ``index`` is the element to add, and the flag determines whether this
//     is to be considered a new addition.

//     Parameters
//     ----------
//     heap: ndarray generated by ``make_heap``
//         The heap object to push into

//     row: int
//         Which actual heap within the heap object to push to

//     weight: float
//         The priority value of the element to push onto the heap

//     index: int
//         The actual value to be pushed

//     flag: int
//         Whether to flag the newly added element || not.

//     Returns
//     -------
//     success: The number of new elements successfully pushed into the heap.
//     """
  static int heapPush(final Heap heap, final int row, final float weight, final int index, final boolean flag) {
    final int[] indices = heap.indices[row];
    final float[] weights = heap.weights[row];
    final boolean[] isNew = heap.isNew[row];

    if (weight >= weights[0]) {
      return 0;
    }

    // break if we already have this element.
    for (int i = 0; i < indices.length; ++i) {
      if (index == indices[i]) {
        return 0;
      }
    }

    // insert val at position zero
    weights[0] = weight;
    indices[0] = index;
    isNew[0] = flag;

    // descend the heap, swapping values until the max heap criterion is met
    int i = 0;
    while (true) {
      final int ic1 = 2 * i + 1;
      final int ic2 = ic1 + 1;
      int iSwap;

      if (ic1 >= heap.indices[0].length) {
        break;
      } else if (ic2 >= heap.indices[0].length) {
        if (weights[ic1] > weight) {
          iSwap = ic1;
        } else {
          break;
        }
      } else if (weights[ic1] >= weights[ic2]) {
        if (weight < weights[ic1]) {
          iSwap = ic1;
        } else {
          break;
        }
      } else {
        if (weight < weights[ic2]) {
          iSwap = ic2;
        } else {
          break;
        }
      }

      weights[i] = weights[iSwap];
      indices[i] = indices[iSwap];
      isNew[i] = isNew[iSwap];

      i = iSwap;
    }

    weights[i] = weight;
    indices[i] = index;
    isNew[i] = flag;

    return 1;
  }


//     Push a new element onto the heap. The heap stores potential neighbors
//     for each data point. The ``row`` parameter determines which data point we
//     are addressing, the ``weight`` determines the distance (for heap sorting),
//     the ``index`` is the element to add, and the flag determines whether this
//     is to be considered a new addition.

//     Parameters
//     ----------
//     heap: ndarray generated by ``make_heap``
//         The heap object to push into

//     row: int
//         Which actual heap within the heap object to push to

//     weight: float
//         The priority value of the element to push onto the heap

//     index: int
//         The actual value to be pushed

//     flag: int
//         Whether to flag the newly added element or not.

//     Returns
//     -------
//     success: The number of new elements successfully pushed into the heap.
  static int uncheckedHeapPush(final Heap heap, final int row, final float weight, final int index, final boolean flag) {
    final int[] indices = heap.indices[row];
    final float[] weights = heap.weights[row];
    final boolean[] isNew = heap.isNew[row];

    if (weight >= weights[0]) {
      return 0;
    }

    // insert val at position zero
    weights[0] = weight;
    indices[0] = index;
    isNew[0] = flag;

    // descend the heap, swapping values until the max heap criterion is met
    int i = 0;
    while (true) {
      final int ic1 = 2 * i + 1;
      final int ic2 = ic1 + 1;

      int iSwap;
      if (ic1 >= heap.indices[0].length) {
        break;
      } else if (ic2 >= heap.indices[0].length) {
        if (weights[ic1] > weight) {
          iSwap = ic1;
        } else {
          break;
        }
      } else if (weights[ic1] >= weights[ic2]) {
        if (weight < weights[ic1]) {
          iSwap = ic1;
        } else {
          break;
        }
      } else {
        if (weight < weights[ic2]) {
          iSwap = ic2;
        } else {
          break;
        }
      }

      weights[i] = weights[iSwap];
      indices[i] = indices[iSwap];
      isNew[i] = isNew[iSwap];

      i = iSwap;
    }

    weights[i] = weight;
    indices[i] = index;
    isNew[i] = flag;

    return 1;
  }


//     Restore the heap property for a heap with an out of place element
//     at position ``elt``. This works with a heap pair where heap1 carries
//     the weights and heap2 holds the corresponding elements.
  private static void siftdown(final float[] heap1, final int[] heap2, int elt) {
    while (elt * 2 + 1 < heap1.length) {
      final int leftChild = elt * 2 + 1;
      final int rightChild = leftChild + 1;
      int swap = elt;

      if (heap1[swap] < heap1[leftChild]) {
        swap = leftChild;
      }

      if (rightChild < heap1.length && heap1[swap] < heap1[rightChild]) {
        swap = rightChild;
      }

      if (swap == elt) {
        break;
      } else {
        final float t = heap1[swap];
        heap1[swap] = heap1[elt];
        heap1[elt] = t;
        final int s = heap2[swap];
        heap2[swap] = heap2[elt];
        heap2[elt] = s;
        elt = swap;
      }
    }
  }


//     Given an array of heaps (of indices and weights), unpack the heap
//     out to give and array of sorted lists of indices and weights by increasing
//     weight. This is effectively just the second half of heap sort (the first
//     half not being required since we already have the data in a heap).
//
//     Parameters
//     ----------
//     heap : array of shape (3, n_samples, n_neighbors)
//         The heap to turn into sorted lists.
//
//     Returns
//     -------
//     indices, weights: arrays of shape (n_samples, n_neighbors)
//         The indices and weights sorted by increasing weight.
//     """
  static Heap deheapSort(Heap heap) {
    int[][] indices = heap.indices;
    float[][] weights = heap.weights;

    for (int i = 0; i < indices.length; ++i) {

      int[] indHeap = indices[i];
      float[] distHeap = weights[i];

      for (int j = 0; j < indHeap.length - 1; ++j) {
        //indHeap[0], indHeap[ indHeap.shape[0] - j - 1 ] = ( indHeap[indHeap.shape[0] - j - 1],   indHeap[0]       );
        int s = indHeap[0];
        indHeap[0] = indHeap[indHeap.length - j - 1];
        indHeap[indHeap.length - j - 1] = s;
        // distHeap[0], distHeap[   distHeap.shape[0] - j - 1  ] = (  distHeap[distHeap.shape[0] - j - 1], distHeap[0]     );
        final float t = distHeap[0];
        distHeap[0] = distHeap[distHeap.length - j - 1];
        distHeap[distHeap.length - j - 1] = t;

        //siftdown(distHeap[:distHeap.shape[0] - j - 1], indHeap[:indHeap.shape[0] - j - 1],  0    );
        siftdown(MathUtils.subarray(distHeap, 0, distHeap.length - j - 1), MathUtils.subarray(indHeap, 0, indHeap.length - j - 1), 0);
      }
    }

    return new Heap(indices, weights);
  }

//     Search the heap for the smallest element that is
//     still flagged.

//     Parameters
//     ----------
//     heap: array of shape (3, n_samples, n_neighbors)
//         The heaps to search

//     row: int
//         Which of the heaps to search

//     Returns
//     -------
//     index: int
//         The index of the smallest flagged element
//         of the ``row``th heap, || -1 if no flagged
//         elements remain in the heap.
  static int smallestFlagged(final Heap heap, final int row) {
    final int[] ind = heap.indices[row];
    final float[] dist = heap.weights[row];
    final boolean[] flag = heap.isNew[row];

    float minDist = Float.POSITIVE_INFINITY;
    int resultIndex = -1;

    for (int i = 0; i < ind.length; ++i) {
      if (flag[i] && dist[i] < minDist) {
        minDist = dist[i];
        resultIndex = i;
      }
    }

    if (resultIndex >= 0) {
      flag[resultIndex] = false;
      return ind[resultIndex];
    } else {
      return -1;
    }
  }


//     Build a heap of candidate neighbors for nearest neighbor descent. For
//     each vertex the candidate neighbors are any current neighbors, and any
//     vertices that have the vertex as one of their nearest neighbors.

//     Parameters
//     ----------
//     current_graph: heap
//         The current state of the graph for nearest neighbor descent.

//     n_vertices: int
//         The total number of vertices in the graph.

//     n_neighbors: int
//         The number of neighbor edges per node in the current graph.

//     max_candidates: int
//         The maximum number of new candidate neighbors.

//     rng_state: array of int64, shape (3,)
//         The internal state of the rng

//     Returns
//     -------
//     candidate_neighbors: A heap with an array of (randomly sorted) candidate
//     neighbors for each vertex in the graph.
  static Heap buildCandidates(final Heap currentGraph, final int nVertices, final int nNeighbors, final int maxCandidates, final Random random) {
    final Heap candidateNeighbors = makeHeap(nVertices, maxCandidates);
    for (int i = 0; i < nVertices; ++i) {
      for (int j = 0; j < nNeighbors; ++j) {
        if (currentGraph.indices[i][j] < 0) {
          continue;
        }
        final int idx = currentGraph.indices[i][j];
        final boolean isn = currentGraph.isNew[i][j];
        final float d = random.nextFloat();
        heapPush(candidateNeighbors, i, d, idx, isn);
        heapPush(candidateNeighbors, idx, d, i, isn);
        currentGraph.isNew[i][j] = false;
      }
    }
    return candidateNeighbors;
  }


// @numba.njit(parallel=true)
// def new_build_candidates(
//     current_graph,
//     n_vertices,
//     n_neighbors,
//     max_candidates,
//     rng_state,
//     rho=0.5,
// ):  # pragma: no cover
//     """Build a heap of candidate neighbors for nearest neighbor descent. For
//     each vertex the candidate neighbors are any current neighbors, and any
//     vertices that have the vertex as one of their nearest neighbors.

//     Parameters
//     ----------
//     current_graph: heap
//         The current state of the graph for nearest neighbor descent.

//     n_vertices: int
//         The total number of vertices in the graph.

//     n_neighbors: int
//         The number of neighbor edges per node in the current graph.

//     max_candidates: int
//         The maximum number of new candidate neighbors.

//     rng_state: array of int64, shape (3,)
//         The internal state of the rng

//     Returns
//     -------
//     candidate_neighbors: A heap with an array of (randomly sorted) candidate
//     neighbors for each vertex in the graph.
//     """
//     new_candidate_neighbors = make_heap(
//         n_vertices, max_candidates
//     )
//     old_candidate_neighbors = make_heap(
//         n_vertices, max_candidates
//     )

//     for i in numba.prange(n_vertices):
//         for j in range(n_neighbors):
//             if current_graph[0, i, j] < 0:
//                 continue
//             idx = current_graph[0, i, j]
//             isn = current_graph[2, i, j]
//             d = tau_rand(rng_state)
//             if tau_rand(rng_state) < rho:
//                 c = 0
//                 if isn:
//                     c += heap_push(
//                         new_candidate_neighbors,
//                         i,
//                         d,
//                         idx,
//                         isn,
//                     )
//                     c += heap_push(
//                         new_candidate_neighbors,
//                         idx,
//                         d,
//                         i,
//                         isn,
//                     )
//                 else:
//                     heap_push(
//                         old_candidate_neighbors,
//                         i,
//                         d,
//                         idx,
//                         isn,
//                     )
//                     heap_push(
//                         old_candidate_neighbors,
//                         idx,
//                         d,
//                         i,
//                         isn,
//                     )

//                 if c > 0:
//                     current_graph[2, i, j] = 0

//     return new_candidate_neighbors, old_candidate_neighbors


//     """Return a submatrix given an original matrix and the indices to keep.
//
//     Parameters
//     ----------
//     mat: array, shape (n_samples, n_samples)
//         Original matrix.
//
//     indicesCol: array, shape (n_samples, nNeighbors)
//         Indices to keep. Each row consists of the indices of the columns.
//
//     nNeighbors: int
//         Number of neighbors.
//
//     Returns
//     -------
//     submat: array, shape (n_samples, nNeighbors)
//         The corresponding submatrix.
//     """
  static float[][] submatrix(float[][] dmat, int[][] indicesCol, int nNeighbors) {
    // todo parallel possible here
    final int nSamplesTransform = dmat.length;
    final int nSamplesFit = dmat[0].length;
    float[][] submat = new float[nSamplesTransform][nNeighbors];
    for (int i = 0; i < nSamplesTransform; ++i) {
      for (int j = 0; j < nNeighbors; ++j) {
        submat[i][j] = dmat[i][indicesCol[i][j]];
      }
    }
    return submat;
  }

  static float[][] submatrix(Matrix dmat, int[][] indicesCol, int nNeighbors) {
    // todo parallel possible here
    // todo speed up for sparse input?
    final int nSamplesTransform = dmat.shape()[0];
    final int nSamplesFit = dmat.shape()[1];
    float[][] submat = new float[nSamplesTransform][nNeighbors];
    for (int i = 0; i < nSamplesTransform; ++i) {
      for (int j = 0; j < nNeighbors; ++j) {
        submat[i][j] = dmat.get(i, indicesCol[i][j]);
      }
    }
    return submat;
  }

  static int[][] submatrix(int[][] dmat, int[][] indicesCol, int nNeighbors) {
    // todo parallel possible here
    final int nSamplesTransform = dmat.length;
    final int nSamplesFit = dmat[0].length;
    int[][] submat = new int[nSamplesTransform][nNeighbors];
    for (int i = 0; i < nSamplesTransform; ++i) {
      for (int j = 0; j < nNeighbors; ++j) {
        submat[i][j] = dmat[i][indicesCol[i][j]];
      }
    }
    return submat;
  }
}
