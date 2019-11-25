package com.tagbio.umap;

// # Author: Leland McInnes <leland.mcinnes@gmail.com>
// #
// # License: BSD 3 clause
// from __future__ import print_function

// import numpy as np
// import numba

// from umap.utils import (
//     tau_rand,
//     make_heap,
//     heap_push,
//     unchecked_heap_push,
//     smallest_flagged,
//     rejection_sample,
//     build_candidates,
//     new_build_candidates,
//     deheap_sort,
// )

// from umap.rp_tree import search_flat_tree

import java.util.Map;

import com.tagbio.umap.metric.Metric;

class NearestNeighborDescent {

  private final Metric dist;
  private final Map<String, Object> dist_args;

//     """Create a numba accelerated version of nearest neighbor descent
//     specialised for the given distance metric and metric arguments. Numba
//     doesn't support higher order functions directly, but we can instead JIT
//     compile the version of NN-descent for any given metric.

//     Parameters
//     ----------
//     dist: function
//         A numba JITd distance function which, given two arrays computes a
//         dissimilarity between them.

//     dist_args: tuple
//         Any extra arguments that need to be passed to the distance function
//         beyond the two arrays to be compared.

//     Returns
//     -------
//     A numba JITd function for nearest neighbor descent computation that is
//     specialised to the given metric.
//     """
  NearestNeighborDescent(final Metric dist, final Map<String, Object> dist_args) {
    this.dist = dist;
    this.dist_args = dist_args;
  }

  Heap nn_descent(final Matrix data, final int nNeighbors, final long[] rng_state, final int maxCandidates, final boolean rp_tree_init, final int n_iters, final int[][] leafArray, final boolean verbose) {
    return nn_descent(data, nNeighbors, rng_state, maxCandidates, rp_tree_init, n_iters, leafArray, verbose, 0.001F, 0.5F);
  }

  Heap nn_descent(final Matrix data, final int nNeighbors, final long[] rng_state, final int maxCandidates, final boolean rp_tree_init, final int n_iters, final int[][] leafArray, final boolean verbose, final float delta, final float rho) {
    final int nVertices = data.shape[0];

    Heap currentGraph = Utils.make_heap(data.shape[0], nNeighbors);
    for (int i = 0; i < data.shape[0]; ++i) {
      final int[] indices = Utils.rejection_sample(nNeighbors, data.shape[0], rng_state);
      for (int j = 0; j < indices.length; ++j) {
        final float d = (float) dist.distance(data.row(i), data.row(indices[j]) /*, dist_args*/);
        Utils.heap_push(currentGraph, i, d, indices[j], true);
        Utils.heap_push(currentGraph, indices[j], d, i, true);
      }
    }
    if (rp_tree_init) {
      for (int n = 0; n < leafArray.length; ++n) {
        for (int i = 0; i < leafArray[n].length; ++i) {
          if (leafArray[n][i] < 0) {
            break;
          }
          for (int j = i + 1; j < leafArray[n].length; ++j) {
            if (leafArray[n][j] < 0) {
              break;
            }
            final float d = (float) dist.distance(data.row(leafArray[n][i]), data.row(leafArray[n][j]) /*,dist_args*/);
            Utils.heap_push(currentGraph, leafArray[n][i], d, leafArray[n][j], true);
            Utils.heap_push(currentGraph, leafArray[n][j], d, leafArray[n][i], true);
          }
        }
      }
    }

    for (int n = 0; n < n_iters; ++n) {
      if (verbose) {
        Utils.message("\t" + n + " / " + n_iters);
      }

      final Heap candidateNeighbors = Utils.build_candidates(currentGraph, nVertices, nNeighbors, maxCandidates, rng_state);

      int c = 0;
      for (int i = 0; i < nVertices; ++i) {
        for (int j = 0; j < maxCandidates; ++j) {
          final int p = candidateNeighbors.indices[i][j];
          if (p < 0 || Utils.tau_rand(rng_state) < rho) {
            continue;
          }
          for (int k = 0; k < maxCandidates; ++k) {
            final int q = candidateNeighbors.indices[i][k];
            if (q < 0 || !candidateNeighbors.isNew[i][j] && !candidateNeighbors.isNew[i][k]) {
              continue;
            }

            final float d = (float) dist.distance(data.row(p), data.row(q) /*, dist_args*/);
            c += Utils.heap_push(currentGraph, p, d, q, true);
            c += Utils.heap_push(currentGraph, q, d, p, true);
          }
        }
      }

      if (c <= delta * nNeighbors * data.shape[0]) {
        break;
      }
    }

    return Utils.deheap_sort(currentGraph);
  }

//    return nn_descent;
//  }


// def make_initialisations(dist, dist_args):
//     @numba.njit(parallel=true)
//     def init_from_random(n_neighbors, data, query_points, heap, rng_state):
//         for i in range(query_points.shape[0]):
//             indices = rejection_sample(n_neighbors, data.shape[0], rng_state)
//             for j in range(indices.shape[0]):
//                 if indices[j] < 0:
//                     continue
//                 d = dist(data[indices[j]], query_points[i], *dist_args)
//                 heap_push(heap, i, d, indices[j], 1)
//         return

//     @numba.njit(parallel=true)
//     def init_from_tree(tree, data, query_points, heap, rng_state):
//         for i in range(query_points.shape[0]):
//             indices = search_flat_tree(
//                 query_points[i],
//                 tree.hyperplanes,
//                 tree.offsets,
//                 tree.children,
//                 tree.indices,
//                 rng_state,
//             )

//             for j in range(indices.shape[0]):
//                 if indices[j] < 0:
//                     continue
//                 d = dist(data[indices[j]], query_points[i], *dist_args)
//                 heap_push(heap, i, d, indices[j], 1)

//         return

//     return init_from_random, init_from_tree


// def initialise_search(
//     forest, data, query_points, n_neighbors, init_from_random, init_from_tree, rng_state
// ):
//     results = make_heap(query_points.shape[0], n_neighbors)
//     init_from_random(n_neighbors, data, query_points, results, rng_state)
//     if forest is not null:
//         for tree in forest:
//             init_from_tree(tree, data, query_points, results, rng_state)

//     return results


// def make_initialized_nnd_search(dist, dist_args):
//     @numba.njit(parallel=true)
//     def initialized_nnd_search(data, indptr, indices, initialization, query_points):

//         for i in numba.prange(query_points.shape[0]):

//             tried = set(initialization[0, i])

//             while true:

//                 # Find smallest flagged vertex
//                 vertex = smallest_flagged(initialization, i)

//                 if vertex == -1:
//                     break
//                 candidates = indices[indptr[vertex] : indptr[vertex + 1]]
//                 for j in range(candidates.shape[0]):
//                     if (
//                         candidates[j] == vertex
//                         || candidates[j] == -1
//                         || candidates[j] in tried
//                     ):
//                         continue
//                     d = dist(data[candidates[j]], query_points[i], *dist_args)
//                     unchecked_heap_push(initialization, i, d, candidates[j], 1)
//                     tried.add(candidates[j])

//         return initialization

//     return initialized_nnd_search
}
