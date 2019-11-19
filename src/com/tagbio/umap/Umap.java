package com.tagbio.umap;

// # Author: Leland McInnes <leland.mcinnes@gmail.com>
// #
// # License: BSD 3 clause

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.tagbio.umap.metric.Metric;
import com.tagbio.umap.metric.ReducedEuclideanMetric;

// """Uniform Manifold Approximation and Projection

// Finds a low dimensional embedding of the data that approximates
// an underlying manifold.

// Parameters
// ----------
// n_neighbors: float (optional, default 15)
//     The size of local neighborhood (in terms of number of neighboring
//     sample points) used for manifold approximation. Larger values
//     result in more global views of the manifold, while smaller
//     values result in more local data being preserved. In general
//     values should be in the range 2 to 100.

// n_components: int (optional, default 2)
//     The dimension of the space to embed into. This defaults to 2 to
//     provide easy visualization, but can reasonably be set to any
//     integer value in the range 2 to 100.

// metric: string || function (optional, default 'euclidean')
//     The metric to use to compute distances in high dimensional space.
//     If a string is passed it must match a valid predefined metric. If
//     a general metric is required a function that takes two 1d arrays and
//     returns a float can be provided. For performance purposes it is
//     required that this be a numba jit'd function. Valid string metrics
//     include:
//         * euclidean
//         * manhattan
//         * chebyshev
//         * minkowski
//         * canberra
//         * braycurtis
//         * mahalanobis
//         * wminkowski
//         * seuclidean
//         * cosine
//         * correlation
//         * haversine
//         * hamming
//         * jaccard
//         * dice
//         * russelrao
//         * kulsinski
//         * rogerstanimoto
//         * sokalmichener
//         * sokalsneath
//         * yule
//     Metrics that take arguments (such as minkowski, mahalanobis etc.)
//     can have arguments passed via the metric_kwds dictionary. At this
//     time care must be taken and dictionary elements must be ordered
//     appropriately; this will hopefully be fixed in the future.

// n_epochs: int (optional, default null)
//     The number of training epochs to be used in optimizing the
//     low dimensional embedding. Larger values result in more accurate
//     embeddings. If null is specified a value will be selected based on
//     the size of the input dataset (200 for large datasets, 500 for small).

// learning_rate: float (optional, default 1.0)
//     The initial learning rate for the embedding optimization.

// init: string (optional, default 'spectral')
//     How to initialize the low dimensional embedding. Options are:
//         * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
//         * 'random': assign initial embedding positions at random.
//         * A numpy array of initial embedding positions.

// min_dist: float (optional, default 0.1)
//     The effective minimum distance between embedded points. Smaller values
//     will result in a more clustered/clumped embedding where nearby points
//     on the manifold are drawn closer together, while larger values will
//     result on a more even dispersal of points. The value should be set
//     relative to the ``spread`` value, which determines the scale at which
//     embedded points will be spread out.

// spread: float (optional, default 1.0)
//     The effective scale of embedded points. In combination with ``min_dist``
//     this determines how clustered/clumped the embedded points are.

// set_op_mix_ratio: float (optional, default 1.0)
//     Interpolate between (fuzzy) union and intersection as the set operation
//     used to combine local fuzzy simplicial sets to obtain a global fuzzy
//     simplicial sets. Both fuzzy set operations use the product t-norm.
//     The value of this parameter should be between 0.0 and 1.0; a value of
//     1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
//     intersection.

// local_connectivity: int (optional, default 1)
//     The local connectivity required -- i.e. the number of nearest
//     neighbors that should be assumed to be connected at a local level.
//     The higher this value the more connected the manifold becomes
//     locally. In practice this should be not more than the local intrinsic
//     dimension of the manifold.

// repulsion_strength: float (optional, default 1.0)
//     Weighting applied to negative samples in low dimensional embedding
//     optimization. Values higher than one will result in greater weight
//     being given to negative samples.

// negative_sample_rate: int (optional, default 5)
//     The number of negative samples to select per positive sample
//     in the optimization process. Increasing this value will result
//     in greater repulsive force being applied, greater optimization
//     cost, but slightly more accuracy.

// transform_queue_size: float (optional, default 4.0)
//     For transform operations (embedding new points using a trained model_
//     this will control how aggressively to search for nearest neighbors.
//     Larger values will result in slower performance but more accurate
//     nearest neighbor evaluation.

// a: float (optional, default null)
//     More specific parameters controlling the embedding. If null these
//     values are set automatically as determined by ``min_dist`` and
//     ``spread``.
// b: float (optional, default null)
//     More specific parameters controlling the embedding. If null these
//     values are set automatically as determined by ``min_dist`` and
//     ``spread``.

// random_state: int, RandomState instance || null, optional (default: null)
//     If int, random_state is the seed used by the random number generator;
//     If RandomState instance, random_state is the random number generator;
//     If null, the random number generator is the RandomState instance used
//     by `np.random`.

// metric_kwds: dict (optional, default null)
//     Arguments to pass on to the metric, such as the ``p`` value for
//     Minkowski distance. If null then no arguments are passed on.

// angular_rp_forest: bool (optional, default false)
//     Whether to use an angular random projection forest to initialise
//     the approximate nearest neighbor search. This can be faster, but is
//     mostly on useful for metric that use an angular style distance such
//     as cosine, correlation etc. In the case of those metrics angular forests
//     will be chosen automatically.

// target_n_neighbors: int (optional, default -1)
//     The number of nearest neighbors to use to construct the target simplcial
//     set. If set to -1 use the ``n_neighbors`` value.

// target_metric: string || callable (optional, default 'categorical')
//     The metric used to measure distance for a target array is using supervised
//     dimension reduction. By default this is 'categorical' which will measure
//     distance in terms of whether categories match || are different. Furthermore,
//     if semi-supervised is required target values of -1 will be trated as
//     unlabelled under the 'categorical' metric. If the target array takes
//     continuous values (e.g. for a regression problem) then metric of 'l1'
//     || 'l2' is probably more appropriate.

// target_metric_kwds: dict (optional, default null)
//     Keyword argument to pass to the target metric when performing
//     supervised dimension reduction. If null then no arguments are passed on.

// target_weight: float (optional, default 0.5)
//     weighting factor between data topology and target topology. A value of
//     0.0 weights entirely on data, a value of 1.0 weights entirely on target.
//     The default of 0.5 balances the weighting equally between data and target.

// transform_seed: int (optional, default 42)
//     Random seed used for the stochastic aspects of the transform operation.
//     This ensures consistency in transform operations.

// verbose: bool (optional, default false)
//     Controls verbosity of logging.
// """
public class Umap {

// from __future__ import print_function
// from warnings import warn
// import time

// from scipy.optimize import curve_fit
// from sklearn.base import BaseEstimator
// from sklearn.utils import check_random_state, check_array
// from sklearn.metrics import pairwise_distances
// from sklearn.preprocessing import normalize
// from sklearn.neighbors import KDTree

// try:
//     import joblib
// except ImportError:
//     # sklearn.externals.joblib is deprecated in 0.21, will be removed in 0.23
//     from sklearn.externals import joblib

// import numpy as np
// import scipy.sparse
// import scipy.sparse.csgraph
// import numba

// import umap.distances as dist

// import umap.sparse as sparse

// from umap.utils import (
//     tau_rand_int,
//     deheap_sort,
//     submatrix,
//     ts,
//     fast_knn_indices,
// )
// from umap.rp_tree import rptree_leaf_array, make_forest
// from umap.nndescent import (
//     make_nn_descent,
//     make_initialisations,
//     make_initialized_nnd_search,
//     initialise_search,
// )
// from umap.spectral import spectral_layout

  private static final int INT32_MIN = Integer.MIN_VALUE + 1;
  private static final int INT32_MAX = Integer.MAX_VALUE - 1;

  private static final double SMOOTH_K_TOLERANCE = 1e-5;
  private static final double MIN_K_DIST_SCALE = 1e-3;
  private static final double NPY_INFINITY = Double.POSITIVE_INFINITY;

  private static Random rng = new Random(42); // todo seed!!!

  /*
    """Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In essence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.

    Parameters
    ----------
    distances: array of shape (n_samples, n_neighbors)
        Distances to nearest neighbors for each samples. Each row should be a
        sorted list of distances to a given samples nearest neighbors.

    k: float
        The number of nearest neighbors to approximate for.

    n_iter: int (optional, default 64)
        We need to binary search for the correct distance value. This is the
        max number of iterations to use in such a search.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    bandwidth: float (optional, default 1)
        The target bandwidth of the kernel, larger values will produce
        larger return values.

    Returns
    -------
    knn_dist: array of shape (n_samples,)
        The distance to kth nearest neighbor, as suitably approximated.

    nn_dist: array of shape (n_samples,)
        The distance to the 1st nearest neighbor for each point.
    """
  */
  private static float[][] smooth_knn_dist(float[][] distances, double k, int n_iter, double local_connectivity, double bandwidth) {
    double target = MathUtils.log2(k) * bandwidth;
    float[] rho = MathUtils.zeros(distances.length);
    float[] result = MathUtils.zeros(distances.length);

    double mean_distances = MathUtils.mean(distances);

    for (int i = 0; i < distances.length; ++i) {
      double lo = 0.0;
      double hi = NPY_INFINITY;
      double mid = 1.0;

      float[] ith_distances = distances[i];
      float[] non_zero_dists = MathUtils.filterPositive(ith_distances);  // ith_distances[ith_distances > 0.0];
      if (non_zero_dists.length >= local_connectivity) {
        int index = (int) Math.floor(local_connectivity);
        double interpolation = local_connectivity - index;
        if (index > 0) {
          rho[i] = non_zero_dists[index - 1];
          if (interpolation > SMOOTH_K_TOLERANCE) {
            rho[i] += interpolation * (non_zero_dists[index] - non_zero_dists[index - 1]);
          }
        } else {
          rho[i] = (float) (interpolation * non_zero_dists[0]);
        }
      } else if (non_zero_dists.length > 0) {
        rho[i] = MathUtils.max(non_zero_dists);
      }

      for (int n = 0; n < n_iter; ++n) {
        double psum = 0.0;
        for (int j = 1; j < distances[0].length; ++j) {   // range(1, distances.shape[1]))
          double d = distances[i][j] - rho[i];
          if (d > 0) {
            psum += Math.exp(-(d / mid));
          } else {
            psum += 1.0;
          }
        }

        if (Math.abs(psum - target) < SMOOTH_K_TOLERANCE) {
          break;
        }

        if (psum > target) {
          hi = mid;
          mid = (lo + hi) / 2.0;
        } else {
          lo = mid;
          if (hi == NPY_INFINITY) {
            mid *= 2;
          } else {
            mid = (lo + hi) / 2.0;
          }
        }
      }

      result[i] = (float) mid;

      if (rho[i] > 0.0) {
        double mean_ith_distances = MathUtils.mean(ith_distances);
        if (result[i] < MIN_K_DIST_SCALE * mean_ith_distances) {
          result[i] = (float) (MIN_K_DIST_SCALE * mean_ith_distances);
        }
      } else {
        if (result[i] < MIN_K_DIST_SCALE * mean_distances) {
          result[i] = (float) (MIN_K_DIST_SCALE * mean_distances);
        }
      }
    }
    return new float[][]{result, rho};
  }

  private static float[][] smooth_knn_dist(float[][] distances, double k) {
    return smooth_knn_dist(distances, k, 64, 1.0, 1.0);
  }

  private static float[][] smooth_knn_dist(float[][] distances, double k, double local_connectivitiy) {
    return smooth_knn_dist(distances, k, 64, local_connectivitiy, 1.0);
  }


  // """Compute the ``n_neighbors`` nearest points for each data point in ``X``
  // under ``metric``. This may be exact, but more likely is approximated via
  // nearest neighbor descent.

  // Parameters
  // ----------
  // X: array of shape (n_samples, n_features)
  //     The input data to compute the k-neighbor graph of.

  // n_neighbors: int
  //     The number of nearest neighbors to compute for each sample in ``X``.

  // metric: string || callable
  //     The metric to use for the computation.

  // metric_kwds: dict
  //     Any arguments to pass to the metric computation function.

  // angular: bool
  //     Whether to use angular rp trees in NN approximation.

  // random_state: np.random state
  //     The random state to use for approximate NN computations.

  // verbose: bool
  //     Whether to print status data during the computation.

  // Returns
  // -------
  // knn_indices: array of shape (n_samples, n_neighbors)
  //     The indices on the ``n_neighbors`` closest points in the dataset.

  // knn_dists: array of shape (n_samples, n_neighbors)
  //     The distances to the ``n_neighbors`` closest points in the dataset.
  // """
  private static Object[] nearest_neighbors(
    Matrix X,
    int n_neighbors,
    Object metric, // either a string or metric
    Map<String, Object> metric_kwds,
    boolean angular,
    long[] random_state, // todo Huh? type conflict // Random
    boolean verbose) {
    if (verbose) {
      Utils.message("Finding Nearest Neighbors");
    }

    int[][] knn_indices;
    float[][] knn_dists;
    List<?> rp_forest;
    if (metric.equals("precomputed")) {
      throw new UnsupportedOperationException();
//      // Note that this does not support sparse distance matrices yet ...
//      // Compute indices of n nearest neighbors
//      knn_indices = Utils.fast_knn_indices(X, n_neighbors);
//      // Compute the nearest neighbor distances
//      //   (equivalent to np.sort(X)[:,:n_neighbors])
//      knn_dists = X[np.arange(X.length())[:,null],knn_indices].copy();
//      rp_forest = [];
    } else {
      Metric distance_func;
      if (metric instanceof Metric) {
        distance_func = (Metric) metric;
        // todo following is complicated due to handling of extra params (e.g. see StandardisedEuclideanMetric)
//      } else if (metric in dist.named_distances) {
//        distance_func = dist.named_distances[metric];
      } else {
        throw new IllegalArgumentException("Metric is neither callable, nor a recognised string");
      }

//      if (metric in (
//                     "cosine",
//                     "correlation",
//                     "dice",
//                     "jaccard",
//                     )) {
//        angular = true;
//      }
      angular = distance_func.isAngular();

      //final long[] rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64);
      final long[] rng_state = new long[]{rng.nextLong(), rng.nextLong(), rng.nextLong()};

      if (X instanceof CsrMatrix) { //scipy.sparse.isspmatrix_csr(X)) {
        final CsrMatrix Y = (CsrMatrix) X;
        if (Sparse.sparse_named_distances.containsKey(metric)){
          distance_func = Sparse.sparse_named_distances.get(metric);
          if (Sparse.sparse_need_n_features.contains(metric)) {
            metric_kwds.put("n_features", X.shape()[1]);
          }
        } else{
          throw new IllegalArgumentException("Metric " + metric + " not supported for sparse data");
        }
        metric_nn_descent = Sparse.make_sparse_nn_descent(distance_func, tuple(metric_kwds.values()));

        int n_trees = 5 + (int) (Math.round(Math.pow(Y.length(), 0.5 / 20.0)));
        int n_iters = Math.max(5, (int) (Math.round(MathUtils.log2(Y.length()))));
        if (verbose) {
          Utils.message("Building RP forest with " + n_trees + " trees");
        }

        rp_forest = RpTree.make_forest(X, n_neighbors, n_trees, rng_state, angular);
        leaf_array = RpTree.rptree_leaf_array(rp_forest);

        if (verbose) {
          Utils.message("NN descent for " + n_iters + " iterations");
        }
        final Object[] nn = NearestNeighborDescent.metric_nn_descent(Y.indices, Y.indptr, Y.data, Y.length(), n_neighbors, rng_state, /*max_candidates=*/60, /*rp_tree_init=*/true,     /*leaf_array=*/leaf_array,  /*n_iters=*/n_iters, verbose);
        knn_indices = (int[][]) nn[0];
        knn_dists = (float[][]) nn[1];
      } else {
        // todo following evilness returns a function to do the nearest neighbour thing
        metric_nn_descent = NearestNeighborDescent.make_nn_descent(distance_func, tuple(metric_kwds.values()));
        int n_trees = 5 + (int) (Math.round(Math.pow(X.length(), 0.5 / 20.0)));
        int n_iters = Math.max(5, (int) (Math.round(MathUtils.log2(X.length()))));

        if (verbose) {
          Utils.message("Building RP forest with " + n_trees + " trees");
        }
        rp_forest = RpTree.make_forest(X, n_neighbors, n_trees, rng_state, angular);
        leaf_array = RpTree.rptree_leaf_array(rp_forest);
        if (verbose) {
          Utils.message("NN descent for " + n_iters + " iterations");
        }
        final Object[] nn = NearestNeighborDescent.metric_nn_descent(X, n_neighbors, rng_state, /*max_candidates=*/60, /*rp_tree_init=*/true, /*leaf_array=*/leaf_array,  /*n_iters=*/n_iters, /*verbose=*/verbose);
        knn_indices = (int[][]) nn[0];
        knn_dists = (float[][]) nn[1];
      }

      if (MathUtils.containsNegative(knn_indices)) {
        Utils.message("Failed to correctly find n_neighbors for some samples. Results may be less than ideal. Try re-running with different parameters.");
      }
    }
    if (verbose) {
      Utils.message("Finished Nearest Neighbor Search");
    }
    return new Object[]{knn_indices, knn_dists, rp_forest};
  }


  // """Construct the membership strength data for the 1-skeleton of each local
  // fuzzy simplicial set -- this is formed as a sparse matrix where each row is
  // a local fuzzy simplicial set, with a membership strength for the
  // 1-simplex to each other data point.

  // Parameters
  // ----------
  // knn_indices: array of shape (n_samples, n_neighbors)
  //     The indices on the ``n_neighbors`` closest points in the dataset.

  // knn_dists: array of shape (n_samples, n_neighbors)
  //     The distances to the ``n_neighbors`` closest points in the dataset.

  // sigmas: array of shape(n_samples)
  //     The normalization factor derived from the metric tensor approximation.

  // rhos: array of shape(n_samples)
  //     The local connectivity adjustment.

  // Returns
  // -------
  // rows: array of shape (n_samples * n_neighbors)
  //     Row data for the resulting sparse matrix (coo format)

  // cols: array of shape (n_samples * n_neighbors)
  //     Column data for the resulting sparse matrix (coo format)

  // vals: array of shape (n_samples * n_neighbors)
  //     Entries for the resulting sparse matrix (coo format)
  // """
  private static Object[] compute_membership_strengths(int[][] knn_indices, float[][] knn_dists, float[] sigmas, float[] rhos) {
    int n_samples = knn_indices.length;
    int n_neighbors = knn_indices[0].length;
    final int size = n_samples * n_neighbors;

    int[] rows = new int[size];
    int[] cols = new int[size];
    float[] vals = new float[size];

    for (int i = 0; i < n_samples; ++i) {
      for (int j = 0; j < n_neighbors; ++j) {
        final double val;
        if (knn_indices[i][j] == -1) {
          continue;  // We didn't get the full knn for i
        }
        if (knn_indices[i][j] == i) {
          val = 0.0;
        } else if (knn_dists[i][j] - rhos[i] <= 0.0) {
          val = 1.0;
        } else {
          val = Math.exp(-((knn_dists[i][j] - rhos[i]) / (sigmas[i])));
        }
        rows[i * n_neighbors + j] = i;
        cols[i * n_neighbors + j] = knn_indices[i][j];
        vals[i * n_neighbors + j] = (float) val;
      }
    }

    return new Object[]{rows, cols, vals};
  }


  // """Given a set of data X, a neighborhood size, and a measure of distance
  // compute the fuzzy simplicial set (here represented as a fuzzy graph in
  // the form of a sparse matrix) associated to the data. This is done by
  // locally approximating geodesic distance at each point, creating a fuzzy
  // simplicial set for each such point, and then combining all the local
  // fuzzy simplicial sets into a global one via a fuzzy union.

  // Parameters
  // ----------
  // X: array of shape (n_samples, n_features)
  //     The data to be modelled as a fuzzy simplicial set.

  // n_neighbors: int
  //     The number of neighbors to use to approximate geodesic distance.
  //     Larger numbers induce more global estimates of the manifold that can
  //     miss finer detail, while smaller values will focus on fine manifold
  //     structure to the detriment of the larger picture.

  // random_state: numpy RandomState || equivalent
  //     A state capable being used as a numpy random state.

  // metric: string || function (optional, default 'euclidean')
  //     The metric to use to compute distances in high dimensional space.
  //     If a string is passed it must match a valid predefined metric. If
  //     a general metric is required a function that takes two 1d arrays and
  //     returns a float can be provided. For performance purposes it is
  //     required that this be a numba jit'd function. Valid string metrics
  //     include:
  //         * euclidean (or l2)
  //         * manhattan (or l1)
  //         * cityblock
  //         * braycurtis
  //         * canberra
  //         * chebyshev
  //         * correlation
  //         * cosine
  //         * dice
  //         * hamming
  //         * jaccard
  //         * kulsinski
  //         * mahalanobis
  //         * matching
  //         * minkowski
  //         * rogerstanimoto
  //         * russellrao
  //         * seuclidean
  //         * sokalmichener
  //         * sokalsneath
  //         * sqeuclidean
  //         * yule
  //         * wminkowski

  //     Metrics that take arguments (such as minkowski, mahalanobis etc.)
  //     can have arguments passed via the metric_kwds dictionary. At this
  //     time care must be taken and dictionary elements must be ordered
  //     appropriately; this will hopefully be fixed in the future.

  // metric_kwds: dict (optional, default {})
  //     Arguments to pass on to the metric, such as the ``p`` value for
  //     Minkowski distance.

  // knn_indices: array of shape (n_samples, n_neighbors) (optional)
  //     If the k-nearest neighbors of each point has already been calculated
  //     you can pass them in here to save computation time. This should be
  //     an array with the indices of the k-nearest neighbors as a row for
  //     each data point.

  // knn_dists: array of shape (n_samples, n_neighbors) (optional)
  //     If the k-nearest neighbors of each point has already been calculated
  //     you can pass them in here to save computation time. This should be
  //     an array with the distances of the k-nearest neighbors as a row for
  //     each data point.

  // angular: bool (optional, default false)
  //     Whether to use angular/cosine distance for the random projection
  //     forest for seeding NN-descent to determine approximate nearest
  //     neighbors.

  // set_op_mix_ratio: float (optional, default 1.0)
  //     Interpolate between (fuzzy) union and intersection as the set operation
  //     used to combine local fuzzy simplicial sets to obtain a global fuzzy
  //     simplicial sets. Both fuzzy set operations use the product t-norm.
  //     The value of this parameter should be between 0.0 and 1.0; a value of
  //     1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
  //     intersection.

  // local_connectivity: int (optional, default 1)
  //     The local connectivity required -- i.e. the number of nearest
  //     neighbors that should be assumed to be connected at a local level.
  //     The higher this value the more connected the manifold becomes
  //     locally. In practice this should be not more than the local intrinsic
  //     dimension of the manifold.

  // verbose: bool (optional, default false)
  //     Whether to report information on the current progress of the algorithm.

  // Returns
  // -------
  // fuzzy_simplicial_set: coo_matrix
  //     A fuzzy simplicial set represented as a sparse matrix. The (i,
  //     j) entry of the matrix represents the membership strength of the
  //     1-simplex between the ith and jth sample points.
  // """
  private static Matrix fuzzy_simplicial_set(
    Matrix X,
    int n_neighbors,
    long[] random_state,
    Object metric,  // todo yuck! fix type
    Map<String, Object> metric_kwds /*={}*/,
    int[][] knn_indices /*=null*/,
    float[][] knn_dists /*=null*/,
    boolean angular /*=false*/,
    float set_op_mix_ratio /*=1.0*/,
    float local_connectivity /*=1.0*/,
    boolean verbose /*=false*/) {
    if (knn_indices == null || knn_dists == null) {
      final Object[] nn = nearest_neighbors(X, n_neighbors, metric, metric_kwds, angular, random_state, verbose);
      knn_indices = (int[][]) nn[0];
      knn_dists = (float[][]) nn[1];
    }

    final float[][] sigmasRhos = smooth_knn_dist(knn_dists, n_neighbors,  /*local_connectivity=*/local_connectivity);
    float[] sigmas = sigmasRhos[0];
    float[] rhos = sigmasRhos[1];

    final Object[] rcv = compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos);
    final int[] rows = (int[]) rcv[0];
    final int[] cols = (int[]) rcv[1];
    final float[] vals = (float[]) rcv[2];

    Matrix result = new CooMatrix(vals, rows, cols, new int[]{(int) X.length(), (int) X.length()});
    result.eliminate_zeros();

    Matrix transpose = result.transpose();

    Matrix prod_matrix = result.multiply(transpose);

    result = result.add(transpose).subtract(prod_matrix).multiply(set_op_mix_ratio).add(prod_matrix.multiply(1.0 - set_op_mix_ratio));

    result.eliminate_zeros();

    return result;
  }


  // """Under the assumption of categorical distance for the intersecting
  // simplicial set perform a fast intersection.

  // Parameters
  // ----------
  // rows: array
  //     An array of the row of each non-zero in the sparse matrix
  //     representation.

  // cols: array
  //     An array of the column of each non-zero in the sparse matrix
  //     representation.

  // values: array
  //     An array of the value of each non-zero in the sparse matrix
  //     representation.

  // target: array of shape (n_samples)
  //     The categorical labels to use in the intersection.

  // unknown_dist: float (optional, default 1.0)
  //     The distance an unknown label (-1) is assumed to be from any point.

  // far_dist float (optional, default 5.0)
  //     The distance between unmatched labels.

  // Returns
  // -------
  // null
  // """
  private static void fast_intersection(int[] rows, int[] cols, float[] values, float[] target, double unknown_dist, double far_dist) {
    for (int nz = 0; nz < rows.length; ++nz) {
      final int i = rows[nz];
      final int j = cols[nz];
      if (target[i] == -1 || target[j] == -1) {
        values[nz] *= Math.exp(-unknown_dist);
      } else if (target[i] != target[j]) {
        values[nz] *= Math.exp(-far_dist);
      }
    }
  }

  private static void fast_intersection(int[] rows, int[] cols, float[] values, float[] target) {
    fast_intersection(rows, cols, values, target, 1.0, 5.0);
  }

  // """Reset the local connectivity requirement -- each data sample should
  // have complete confidence in at least one 1-simplex in the simplicial set.
  // We can enforce this by locally rescaling confidences, and then remerging the
  // different local simplicial sets together.

  // Parameters
  // ----------
  // simplicial_set: sparse matrix
  //     The simplicial set for which to recalculate with respect to local
  //     connectivity.

  // Returns
  // -------
  // simplicial_set: sparse_matrix
  //     The recalculated simplicial set, now with the local connectivity
  //     assumption restored.
  // """
  private static Matrix reset_local_connectivity(Matrix simplicial_set) {
    simplicial_set = Normalize.normalize(simplicial_set, "max");
    Matrix transpose = simplicial_set.transpose();
    Matrix prod_matrix = simplicial_set.multiply(transpose);
    simplicial_set = simplicial_set.add(transpose).subtract(prod_matrix);
    simplicial_set.eliminate_zeros();

    return simplicial_set;
  }

  // """Combine a fuzzy simplicial set with another fuzzy simplicial set
  // generated from categorical data using categorical distances. The target
  // data is assumed to be categorical label data (a vector of labels),
  // and this will update the fuzzy simplicial set to respect that label data.

  // TODO: optional category cardinality based weighting of distance

  // Parameters
  // ----------
  // simplicial_set: sparse matrix
  //     The input fuzzy simplicial set.

  // target: array of shape (n_samples)
  //     The categorical labels to use in the intersection.

  // unknown_dist: float (optional, default 1.0)
  //     The distance an unknown label (-1) is assumed to be from any point.

  // far_dist float (optional, default 5.0)
  //     The distance between unmatched labels.

  // Returns
  // -------
  // simplicial_set: sparse matrix
  //     The resulting intersected fuzzy simplicial set.
  // """
  private static Matrix categorical_simplicial_set_intersection(CooMatrix simplicial_set, float[] target, double unknown_dist, double far_dist/*=5.0*/) {
    simplicial_set = simplicial_set.tocoo();

    fast_intersection(
      simplicial_set.row,
      simplicial_set.col,
      simplicial_set.data,
      target,
      unknown_dist,
      far_dist);

    simplicial_set.eliminate_zeros();

    return reset_local_connectivity(simplicial_set);
  }

  private static Matrix categorical_simplicial_set_intersection(CooMatrix simplicial_set, float[] target, double far_dist/*=5.0*/) {
    return categorical_simplicial_set_intersection(simplicial_set, target, 1.0, far_dist);
  }

  private static Matrix general_simplicial_set_intersection(Matrix simplicial_set1, Matrix simplicial_set2, float weight) {

    CooMatrix result = simplicial_set1.add(simplicial_set2).tocoo();
    CsrMatrix left = simplicial_set1.tocsr();
    CsrMatrix right = simplicial_set2.tocsr();

    Sparse.general_sset_intersection(
      left.indptr,
      left.indices,
      left.data,
      right.indptr,
      right.indices,
      right.data,
      result.row,
      result.col,
      result.data,
      weight);

    return result;
  }

  // """Given a set of weights and number of epochs generate the number of
  // epochs per sample for each weight.

  // Parameters
  // ----------
  // weights: array of shape (n_1_simplices)
  //     The weights of how much we wish to sample each 1-simplex.

  // n_epochs: int
  //     The total number of epochs we want to train for.

  // Returns
  // -------
  // An array of number of epochs per sample, one for each 1-simplex.
  // """
  private static float[] make_epochs_per_sample(float[] weights, int n_epochs) {
    final float[] result = new float[weights.length]; //-1.0 * np.ones(weights.length, dtype=np.float64);
    Arrays.fill(result, -1.0F);
    final float[] n_samples = MathUtils.multiply(MathUtils.divide(weights, MathUtils.max(weights)), n_epochs);
    //result[n_samples > 0] = (float) (n_epochs) / n_samples[n_samples > 0]);
    for (int k = 0; k < n_samples.length; ++k) {
      if (n_samples[k] > 0) {
        result[k] = n_epochs / n_samples[k];
      }
    }
    return result;
  }


  // """Standard clamping of a value into a fixed range (in this case -4.0 to
  // 4.0)

  // Parameters
  // ----------
  // val: float
  //     The value to be clamped.

  // Returns
  // -------
  // The clamped value, now fixed to be in the range -4.0 to 4.0.
  // """
  private static double clip(double val) {
    if (val > 4.0) {
      return 4.0;
    } else if (val < -4.0) {
      return -4.0;
    } else {
      return val;
    }
  }


  // """Reduced Euclidean distance.
  // Parameters
  // ----------
  // x: array of shape (embedding_dim,)
  // y: array of shape (embedding_dim,)

  // Returns
  // -------
  // The squared euclidean distance between x and y
  // """
  private static double rdist(float[] x, float[] y) {
    return ReducedEuclideanMetric.SINGLETON.distance(x, y);
  }


  // """Improve an embedding using stochastic gradient descent to minimize the
  // fuzzy set cross entropy between the 1-skeletons of the high dimensional
  // and low dimensional fuzzy simplicial sets. In practice this is done by
  // sampling edges based on their membership strength (with the (1-p) terms
  // coming from negative sampling similar to word2vec).

  // Parameters
  // ----------
  // head_embedding: array of shape (n_samples, n_components)
  //     The initial embedding to be improved by SGD.

  // tail_embedding: array of shape (source_samples, n_components)
  //     The reference embedding of embedded points. If not embedding new
  //     previously unseen points with respect to an existing embedding this
  //     is simply the head_embedding (again); otherwise it provides the
  //     existing embedding to embed with respect to.

  // head: array of shape (n_1_simplices)
  //     The indices of the heads of 1-simplices with non-zero membership.

  // tail: array of shape (n_1_simplices)
  //     The indices of the tails of 1-simplices with non-zero membership.

  // n_epochs: int
  //     The number of training epochs to use in optimization.

  // n_vertices: int
  //     The number of vertices (0-simplices) in the dataset.

  // epochs_per_samples: array of shape (n_1_simplices)
  //     A float value of the number of epochs per 1-simplex. 1-simplices with
  //     weaker membership strength will have more epochs between being sampled.

  // a: float
  //     Parameter of differentiable approximation of right adjoint functor

  // b: float
  //     Parameter of differentiable approximation of right adjoint functor

  // rng_state: array of int64, shape (3,)
  //     The internal state of the rng

  // gamma: float (optional, default 1.0)
  //     Weight to apply to negative samples.

  // initial_alpha: float (optional, default 1.0)
  //     Initial learning rate for the SGD.

  // negative_sample_rate: int (optional, default 5)
  //     Number of negative samples to use per positive sample.

  // verbose: bool (optional, default false)
  //     Whether to report information on the current progress of the algorithm.

  // Returns
  // -------
  // embedding: array of shape (n_samples, n_components)
  //     The optimized embedding.
  // """
  private static Matrix optimize_layout(
    Matrix head_embedding,
    Matrix tail_embedding,
    int[] head,
    int[] tail,
    int n_epochs,
    int n_vertices,
    float[] epochs_per_sample,
    float a,
    float b,
    long[] rng_state,
    float gamma /*=1.0*/,
    double initial_alpha /*=1.0*/,
    float negative_sample_rate /*=5.0*/,
    boolean verbose) {

    int dim = head_embedding.shape()[1];
    boolean move_other = (head_embedding.shape()[0] == tail_embedding.shape()[0]);
    double alpha = initial_alpha;

    float[] epochs_per_negative_sample = MathUtils.divide(epochs_per_sample, negative_sample_rate);
    float[] epoch_of_next_negative_sample = Arrays.copyOf(epochs_per_negative_sample, epochs_per_negative_sample.length);
    float[] epoch_of_next_sample = Arrays.copyOf(epochs_per_sample, epochs_per_sample.length);

    for (int n = 0; n < n_epochs; ++n) {
      for (int i = 0; i < epochs_per_sample.length; ++i) {
        if (epoch_of_next_sample[i] <= n) {
          int j = head[i];
          int k = tail[i];

          float[] current = head_embedding.row(j);
          float[] other = tail_embedding.row(k);

          double dist_squared = rdist(current, other);

          double grad_coeff;
          if (dist_squared > 0.0) {
            grad_coeff = (-2.0 * a * b * Math.pow(dist_squared, b - 1.0));
            grad_coeff /= (a * Math.pow(dist_squared, b) + 1.0);
          } else {
            grad_coeff = 0.0;
          }

          for (int d = 0; d < dim; ++d) {
            final double grad_d = clip(grad_coeff * (current[d] - other[d]));
            current[d] += grad_d * alpha;
            if (move_other) {
              other[d] += -grad_d * alpha;
            }
          }

          epoch_of_next_sample[i] += epochs_per_sample[i];

          int n_neg_samples = (int) ((n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]);

          for (int p = 0; p < n_neg_samples; ++p) {
            k = Utils.tau_rand_int(rng_state) % n_vertices;

            other = tail_embedding.row(k);

            dist_squared = rdist(current, other);

            if (dist_squared > 0.0) {
              grad_coeff = 2.0 * gamma * b;
              grad_coeff /= (0.001 + dist_squared) * (a * Math.pow(dist_squared, b) + 1);
            } else if (j == k) {
              continue;
            } else {
              grad_coeff = 0.0;
            }

            for (int d = 0; d < dim; ++d) {
              final double grad_d;
              if (grad_coeff > 0.0) {
                grad_d = clip(grad_coeff * (current[d] - other[d]));
              } else {
                grad_d = 4.0;
              }
              current[d] += grad_d * alpha;
            }
          }

          epoch_of_next_negative_sample[i] += (n_neg_samples * epochs_per_negative_sample[i]);
        }
      }

      alpha = initial_alpha * (1.0 - (float) (n) / (float) (n_epochs));

      if (verbose && n % (n_epochs / 10) == 0) {
        Utils.message("\tcompleted " + n + "/" + n_epochs + " epochs");
      }
    }
    return head_embedding;
  }


  // """Perform a fuzzy simplicial set embedding, using a specified
  // initialisation method and then minimizing the fuzzy set cross entropy
  // between the 1-skeletons of the high and low dimensional fuzzy simplicial
  // sets.

  // Parameters
  // ----------
  // data: array of shape (n_samples, n_features)
  //     The source data to be embedded by UMAP.

  // graph: sparse matrix
  //     The 1-skeleton of the high dimensional fuzzy simplicial set as
  //     represented by a graph for which we require a sparse matrix for the
  //     (weighted) adjacency matrix.

  // n_components: int
  //     The dimensionality of the euclidean space into which to embed the data.

  // initial_alpha: float
  //     Initial learning rate for the SGD.

  // a: float
  //     Parameter of differentiable approximation of right adjoint functor

  // b: float
  //     Parameter of differentiable approximation of right adjoint functor

  // gamma: float
  //     Weight to apply to negative samples.

  // negative_sample_rate: int (optional, default 5)
  //     The number of negative samples to select per positive sample
  //     in the optimization process. Increasing this value will result
  //     in greater repulsive force being applied, greater optimization
  //     cost, but slightly more accuracy.

  // n_epochs: int (optional, default 0)
  //     The number of training epochs to be used in optimizing the
  //     low dimensional embedding. Larger values result in more accurate
  //     embeddings. If 0 is specified a value will be selected based on
  //     the size of the input dataset (200 for large datasets, 500 for small).

  // init: string
  //     How to initialize the low dimensional embedding. Options are:
  //         * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
  //         * 'random': assign initial embedding positions at random.
  //         * A numpy array of initial embedding positions.

  // random_state: numpy RandomState || equivalent
  //     A state capable being used as a numpy random state.

  // metric: string
  //     The metric used to measure distance in high dimensional space; used if
  //     multiple connected components need to be layed out.

  // metric_kwds: dict
  //     Key word arguments to be passed to the metric function; used if
  //     multiple connected components need to be layed out.

  // verbose: bool (optional, default false)
  //     Whether to report information on the current progress of the algorithm.

  // Returns
  // -------
  // embedding: array of shape (n_samples, n_components)
  //     The optimized of ``graph`` into an ``n_components`` dimensional
  //     euclidean space.
  // """
  private static Matrix simplicial_set_embedding(
    Matrix data,
    CooMatrix graph,
    int n_components,
    double initial_alpha,
    float a,
    float b,
    float gamma,
    int negative_sample_rate,
    int n_epochs,
    String init,
    long[] random_state,
    Object metric, // todo yuck
    Map<String, Object> metric_kwds,
    boolean verbose) {
    graph = graph.tocoo();
    graph.sum_duplicates();
    int n_vertices = graph.shape()[1];

    if (n_epochs <= 0) {
      // For smaller datasets we can use more epochs
      if (graph.length() <= 10000) {
        n_epochs = 500;
      } else {
        n_epochs = 200;
      }
    }

    MathUtils.zeroEntriesBelowLimit(graph.data, MathUtils.max(graph.data) / (float) n_epochs);
    graph.eliminate_zeros();

    Matrix embedding;
    if (init instanceof String && init.equals("random")) {
      embedding = random_state.uniform(low = -10.0, high = 10.0, size = (graph.length(), n_components)).astype(np.float32);
    } else if (init instanceof String && init.equals("spectral")) {
      // We add a little noise to avoid local minima for optimization to come
      float[][] initialisation = Spectral.spectral_layout(data, graph, n_components, random_state, /*metric=*/metric, /*metric_kwds=*/metric_kwds);
      float expansion = 10.0 / Math.abs(initialisation).max();
      embedding = (MathUtils.multiply(initialisation, expansion)).astype(np.float32) + random_state.normal(scale = 0.0001, size =[graph.length(), n_components]).astype(np.float32);
    } else {
      // Situation where init contains prepared data
      throw new UnsupportedOperationException();
//      init_data = np.array(init);
//      if (len(init_data.shape) == 2) {
//        if (np.unique(init_data, /*axis =*/ 0).length < init_data.length) {
//          tree = KDTree(init_data);
//          float[][] dist /*, ind*/ = tree.query(init_data, k = 2);
//          double nndist = MathUtils.mean(dist, 1);
//          embedding = init_data + random_state.normal(scale = 0.001 * nndist, size = init_data.shape).astype(np.float32);
//        } else {
//          embedding = init_data;
//        }
//      }
    }

    final float[] epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs);
    int[] head = graph.row;
    int[] tail = graph.col;

    //long[]  rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64);
    final long[] rng_state = new long[]{rng.nextLong(), rng.nextLong(), rng.nextLong()};
    Matrix res_embedding = optimize_layout(
      embedding,
      embedding,
      head,
      tail,
      n_epochs,
      n_vertices,
      epochs_per_sample,
      a,
      b,
      rng_state,
      gamma,
      initial_alpha,
      negative_sample_rate,
      verbose);

    return res_embedding;
  }


  // """Given indices and weights and an original embeddings
  // initialize the positions of new points relative to the
  // indices and weights (of their neighbors in the source data).

  // Parameters
  // ----------
  // indices: array of shape (n_new_samples, n_neighbors)
  //     The indices of the neighbors of each new sample

  // weights: array of shape (n_new_samples, n_neighbors)
  //     The membership strengths of associated 1-simplices
  //     for each of the new samples.

  // embedding: array of shape (n_samples, dim)
  //     The original embedding of the source data.

  // Returns
  // -------
  // new_embedding: array of shape (n_new_samples, dim)
  //     An initial embedding of the new sample points.
  // """
  private static Matrix init_transform(int[][] indices, float[][] weights, Matrix embedding) {
    final float[][] result = new float[indices.length][embedding.shape()[1]];
    for (int i = 0; i < indices.length; ++i) {
      for (int j = 0; j < indices[i].length; ++j) {
        for (int d = 0; d < embedding.shape()[1]; ++d) {
          result[i][d] += (weights[i][j] * embedding.get(indices[i][j], d));
        }
      }
    }

    return new DefaultMatrix(result);
  }

  private static double curve(final float x, final float a, final float b) {
    return 1.0 / (1.0 + a * Math.pow(x, 2 * b));
  }

  // """Fit a, b params for the differentiable curve used in lower
  // dimensional fuzzy simplicial complex construction. We want the
  // smooth curve (from a pre-defined family with simple gradient) that
  // best matches an offset exponential decay.
  // """
  private static float[] find_ab_params(float spread, float min_dist) {
    /*
    float[] xv = MathUtils.linspace(0, spread * 3, 300);
    float[] yv = new float[xv.length];
    //  yv[xv < min_dist] = 1.0;
    //  yv[xv >= min_dist] = Math.exp(-(xv[xv >= min_dist] - min_dist) / spread   );
    for (int k = 0; k < yv.length; ++k) {
      if (xv[k] < min_dist) {
        yv[k] = 1.0F;
      } else {
        yv[k] = (float) Math.exp(-(xv[k] - min_dist) / spread);
      }
    }
    final float[] params = curve_fit(curve, xv, yv); // todo here curve is the function above -- curve_fit in scipy
    return new float[]{params[0], params[1]};
    */
    if (spread == 1.0F && min_dist == 0.1F) {
      return new float[]{1.57694346F, 0.89506088F};
    }
    throw new UnsupportedOperationException();
  }

  private boolean angular_rp_forest = false;
  private String init = "spectral";
  private int n_neighbors = 15;
  private int n_components = 2;
  private Integer n_epochs = null;
  private Object metric = "euclidean";
  private final Map<String, String> metric_kwds = new HashMap<>();
  private float learning_rate = 1.0F;
  private float repulsion_strength = 1.0F;
  private float min_dist = 0.1F;
  private float spread = 1.0F;
  private float set_op_mix_ratio = 1.0F;
  private float local_connectivity = 1.0F;
  private int negative_sample_rate = 5;
  private float transform_queue_size = 4.0F;
  private Object target_metric = "categorical"; // todo would prefer this to be type Metric
  private int target_n_neighbors = -1;
  private float target_weight = 0.5F;
  private int transform_seed = 42;
  private final Map<String, Object> target_metric_kwds = new HashMap<>();
  private boolean verbose = false;
  private Float a = null;
  private Float b = null;
  private long[] random_state = new long[] {rng.nextLong(), rng.nextLong(), rng.nextLong()};

  private double _initial_alpha;
  private int _n_neighbors;
  private boolean _sparse_data;
  private float _a;
  private float _b;
  private HashMap<String, Object> _metric_kwds;
  private HashMap<String, Object> _target_metric_kwds;
  private Matrix _raw_data;
  private CsrMatrix _search_graph;
  private int[][] _knn_indices;
  private float[][] _knn_dists;
  private List<?> _rp_forest;
  private boolean _small_data;
  private Metric _distance_func;
  private Matrix graph_;
  private Matrix embedding_;

  public void setInit(final String init) {
    this.init = init;
  }

  public void setNumberNearestNeighbours(final int n_neighbors) {
    this.n_neighbors = n_neighbors;
  }

  public void setNumberComponents(final int n_components) {
    this.n_components = n_components;
  }

  public void setNumberEpochs(final int n_epochs) {
    this.n_epochs = n_epochs;
  }

  public void setMetric(final Metric metric) {
    this.metric = metric;
  }

  public void setMetric(final String metric) {
    this.metric = metric;
  }

  public void clearMetricKeywords() {
    metric_kwds.clear();
  }

  public void addMetricKeyword(final String key, final String value) {
    metric_kwds.put(key, value);
  }

  public void clearTargetMetricKeywords() {
    target_metric_kwds.clear();
  }

  public void addTargetMetricKeyword(final String key, final String value) {
    target_metric_kwds.put(key, value);
  }

  public void setLearningRate(final float rate) {
    learning_rate = rate;
  }

  public void setRepulsionStrength(final float repulsion_strength) {
    this.repulsion_strength = repulsion_strength;
  }

  public void setMinDist(final float min_dist) {
    this.min_dist = min_dist;
  }

  public void setSpread(final float spread) {
    this.spread = spread;
  }

  public void setSet_op_mix_ratio(final float set_op_mix_ratio) {
    this.set_op_mix_ratio = set_op_mix_ratio;
  }

  public void setLocalConnectivity(final float local_connectivity) {
    this.local_connectivity = local_connectivity;
  }

  public void setNegativeSampleRate(final int negative_sample_rate) {
    this.negative_sample_rate = negative_sample_rate;
  }

  public void setTarget_metric(final String target_metric) {
    this.target_metric = target_metric;
  }

  public void setVerbose(final boolean verbose) {
    this.verbose = verbose;
  }

  public void setRandom_state(final long[] random_state) {
    this.random_state = random_state;
  }

  public void setTransformQueueSize(final float transform_queue_size) {
    this.transform_queue_size = transform_queue_size;
  }

  public void setAngularRpForest(final boolean angular_rp_forest) {
    this.angular_rp_forest = angular_rp_forest;
  }

  public void setTarget_n_neighbors(final int target_n_neighbors) {
    this.target_n_neighbors = target_n_neighbors;
  }

  public void setTargetWeight(final float target_weight) {
    this.target_weight = target_weight;
  }

  public void setTransformSeed(final int transform_seed) {
    this.transform_seed = transform_seed;
  }

  private void _validate_parameters() {
    if (set_op_mix_ratio < 0.0 || set_op_mix_ratio > 1.0) {
      throw new IllegalArgumentException("set_op_mix_ratio must be between 0.0 and 1.0");
    }
    if (repulsion_strength < 0.0) {
      throw new IllegalArgumentException("repulsion_strength cannot be negative");
    }
    if (min_dist > spread) {
      throw new IllegalArgumentException("min_dist must be less than || equal to spread");
    }
    if (min_dist < 0.0) {
      throw new IllegalArgumentException("min_dist must be greater than 0.0");
    }
//    if (!isinstance(init, str) && !isinstance(init, np.ndarray)) {
//      throw new IllegalArgumentException("init must be a string || ndarray");
//    }
    if (init instanceof String && (!"spectral".equals(init) && !"random".equals(init))) {
      throw new IllegalArgumentException("string init values must be 'spectral' || 'random'");
    }
//    if (isinstance(init, np.ndarray) && init.shape[1] != n_components) {
//      throw new IllegalArgumentException("init ndarray must match n_components value");
//    }
    if (!(metric instanceof String) && !(metric instanceof Metric)) {
      throw new IllegalArgumentException("metric must be string || callable");
    }
    if (negative_sample_rate < 0) {
      throw new IllegalArgumentException("negative sample rate must be positive");
    }
    if (_initial_alpha < 0.0) {
      throw new IllegalArgumentException("learning_rate must be positive");
    }
    if (n_neighbors < 2) {
      throw new IllegalArgumentException("n_neighbors must be greater than 2");
    }
    if (target_n_neighbors < 2 && target_n_neighbors != -1) {
      throw new IllegalArgumentException("target_n_neighbors must be greater than 2");
    }
    //        if (! isinstance(this.n_components, int)) {
    //          throw new IllegalArgumentException("n_components must be an int");
    //        }
    if (n_components < 1) {
      throw new IllegalArgumentException("n_components must be greater than 0");
    }
    if (n_epochs != null && n_epochs <= 10) {
      throw new IllegalArgumentException("n_epochs must be a positive integer larger than 10");
    }
  }

  // """Fit X into an embedded space.

  // Optionally use y for supervised dimension reduction.

  // Parameters
  // ----------
  // X : array, shape (n_samples, n_features) || (n_samples, n_samples)
  //     If the metric is 'precomputed' X must be a square distance
  //     matrix. Otherwise it contains a sample per row. If the method
  //     is 'exact', X may be a sparse matrix of type 'csr', 'csc'
  //     || 'coo'.

  // y : array, shape (n_samples)
  //     A target array for supervised dimension reduction. How this is
  //     handled is determined by parameters UMAP was instantiated with.
  //     The relevant attributes are ``target_metric`` and
  //     ``target_metric_kwds``.
  // """
  private void fit(Matrix X, float[] y /*=null*/) {

    //X = check_array(X, dtype = np.float32, accept_sparse = "csr");
    this._raw_data = X;

    // Handle all the optional arguments, setting default
    if (this.a == null || this.b == null) {
      final float[] ab = find_ab_params(this.spread, this.min_dist);
      this._a = ab[0];
      this._b = ab[1];
    } else {
      this._a = this.a;
      this._b = this.b;

      this._metric_kwds = new HashMap<>(this.metric_kwds);
      this._target_metric_kwds = new HashMap<>(this.target_metric_kwds);

//      if (isinstance(this.init, np.ndarray)) {
//        init = check_array(this.init,        /*  dtype = */np.float32,         /* accept_sparse =*/ false);
//      } else {
//        init = this.init;
//      }
      String init = this.init;

      this._initial_alpha = this.learning_rate;

      this._validate_parameters();

      if (this.verbose) {
        Utils.message(this.toString()); // todo?  Huh? will this do anything useful -- there is no toString()
      }

      // Error check n_neighbors based on data size
      if (X.length() <= this.n_neighbors) {
        if (X.length() == 1) {
          this.embedding_ = new DefaultMatrix(new float[1][this.n_components]); // MathUtils.zeros((1, this.n_components) );  // needed to sklearn comparability
          return;
        }

        Utils.message("n_neighbors is larger than the dataset size; truncating to X.length - 1");
        this._n_neighbors = (int) (X.length() - 1);
      } else {
        this._n_neighbors = this.n_neighbors;
      }

      if (X instanceof CsrMatrix) {   // scipy.sparse.isspmatrix_csr(X)) {
        final CsrMatrix Y = (CsrMatrix) X;
        if (!Y.has_sorted_indices()) {
          Y.sort_indices();
        }
        this._sparse_data = true;
      } else {
        this._sparse_data = false;
      }

      long[] random_state = this.random_state; //check_random_state(this.random_state);

      if (this.verbose) {
        Utils.message("Construct fuzzy simplicial set");
      }

      // Handle small cases efficiently by computing all distances
      if (X.length() < 4096) {
        this._small_data = true;
        Matrix dmat = PairwiseDistances.pairwise_distances(X, (Metric) this.metric, this._metric_kwds);
        this.graph_ = fuzzy_simplicial_set(
          dmat,
          this._n_neighbors,
          random_state,
          "precomputed",
          this._metric_kwds,
          null,
          null,
          this.angular_rp_forest,
          this.set_op_mix_ratio,
          this.local_connectivity,
          this.verbose
        );
      } else {
        this._small_data = false;
        // Standard case
        final Object[] nn = nearest_neighbors(X, this._n_neighbors, this.metric, this._metric_kwds, this.angular_rp_forest, random_state, this.verbose);

          this._knn_indices = (int[][]) nn[0];
          this._knn_dists = (float[][]) nn[1];
          this._rp_forest = (List<?>) nn[2];

        this.graph_ = fuzzy_simplicial_set(
          X,
          this.n_neighbors,
          random_state,
          this.metric,
          this._metric_kwds,
          this._knn_indices,
          this._knn_dists,
          this.angular_rp_forest,
          this.set_op_mix_ratio,
          this.local_connectivity,
          this.verbose
        );

        // todo this starts as LilMatrix type but ends up as a CsrMatrix!
        // todo according to scipy an efficiency thing -- but bytes ??
        this._search_graph = scipy.sparse.lil_matrix((X.length(), X.length()), dtype = np.int8      );
        this._search_graph.rows = this._knn_indices;
        this._search_graph.data = (this._knn_dists != 0).astype(np.int8);
        this._search_graph = this._search_graph.maximum(      this._search_graph.transpose()   ).tocsr();

        if (this.metric instanceof Metric) {
          this._distance_func = (Metric) this.metric;
//        } else if (this.metric in dist.named_distances) {
//          this._distance_func = dist.named_distances[this.metric];
        } else if (this.metric == "precomputed") {
          Utils.message("Using precomputed metric; transform will be unavailable for new data");
        } else {
          throw new IllegalArgumentException("Metric is neither callable, nor a recognised string");
        }

        if (!"precomputed".equals(this.metric)) {
          this._dist_args = tuple(this._metric_kwds.values()); // todo this is weird? -- how do values get associated with what they are?
          this._random_init, this._tree_init = make_initialisations(this._distance_func, this._dist_args);
          this._search = make_initialized_nnd_search(this._distance_func, this._dist_args);
        }
      }
    }


    if (y != null) {
      if (X.length() != y.length) {
        throw new IllegalArgumentException("Length of x =  " + X.length() + ", length of y = " + y.length + ", while it must be equal.");
      }
      //float[] y_ = check_array(y, ensure_2d = false);
      float[] y_ = y;
      if (this.target_metric.equals("categorical")) {
        final double far_dist;
        if (this.target_weight < 1.0) {
          far_dist = 2.5 * (1.0 / (1.0 - this.target_weight));
        } else {
          far_dist = 1.0e12;
        }
        this.graph_ = categorical_simplicial_set_intersection((CooMatrix) this.graph_, y_, /*far_dist =*/ far_dist);
      } else {
        final int target_n_neighbors;
        if (this.target_n_neighbors == -1) {
          target_n_neighbors = this._n_neighbors;
        } else {
          target_n_neighbors = (this.target_n_neighbors);
        }

        Matrix target_graph;
        // Handle the small case as precomputed as before
        if (y.length < 4096) {
          //Matrix ydmat = PairwiseDistances.pairwise_distances(y_[np.newaxis, :].T,  (Metric) this.target_metric,  this._target_metric_kwds);
          Matrix ydmat = PairwiseDistances.pairwise_distances(MathUtils.promoteTranspose(y_),  (Metric) this.target_metric,  this._target_metric_kwds);
          target_graph = fuzzy_simplicial_set(
            ydmat,
            target_n_neighbors,
            random_state,
            "precomputed",
            this._target_metric_kwds,
            null,
            null,
            false,
            1.0F,
            1.0F,
            false);
        } else {
          // Standard case
          target_graph = fuzzy_simplicial_set(
            MathUtils.promoteTranspose(y_),  //y_[np.newaxis, :].T,
            target_n_neighbors,
            random_state,
            this.target_metric,
            this._target_metric_kwds,
            null,
            null,
            false,
            1.0F,
            1.0F,
            false);
        }
        // product = this.graph_.multiply(target_graph)
        // // this.graph_ = 0.99 * product + 0.01 * (this.graph_ +
        // //                                        target_graph -
        // //                                        product)
        // this.graph_ = product
        this.graph_ = general_simplicial_set_intersection(this.graph_, target_graph, this.target_weight);
        this.graph_ = reset_local_connectivity(this.graph_);
      }
    }


    int n_epochs;
    if (this.n_epochs == null) {
      n_epochs = 0;
    } else {
      n_epochs = this.n_epochs;
    }

    if (this.verbose) {
      Utils.message("Construct embedding");
    }

    this.embedding_ = simplicial_set_embedding(
      this._raw_data,
      (CooMatrix) this.graph_,
      this.n_components,
      this._initial_alpha,
      this._a,
      this._b,
      this.repulsion_strength,
      this.negative_sample_rate,
      n_epochs,
      init,
      random_state,
      this.metric,
      this._metric_kwds,
      this.verbose
    );

    if (this.verbose) {
      Utils.message("Finished embedding");
    }

    //this._input_hash = joblib.hash(this._raw_data);

    //return this;
  }

  // """Fit X into an embedded space and return that transformed
  // output.

  // Parameters
  // ----------
  // X : array, shape (n_samples, n_features) || (n_samples, n_samples)
  //     If the metric is 'precomputed' X must be a square distance
  //     matrix. Otherwise it contains a sample per row.

  // y : array, shape (n_samples)
  //     A target array for supervised dimension reduction. How this is
  //     handled is determined by parameters UMAP was instantiated with.
  //     The relevant attributes are ``target_metric`` and
  //     ``target_metric_kwds``.

  // Returns
  // -------
  // X_new : array, shape (n_samples, n_components)
  //     Embedding of the training data in low-dimensional space.
  // """
  Matrix fit_transform(Matrix X, float[] y) {
    fit(X, y);
    return this.embedding_;
  }

  Matrix fit_transform(Matrix X) {
    return fit_transform(X, null);
  }

  // """Transform X into the existing embedded space and return that
  // transformed output.

  // Parameters
  // ----------
  // X : array, shape (n_samples, n_features)
  //     New data to be transformed.

  // Returns
  // -------
  // X_new : array, shape (n_samples, n_components)
  //     Embedding of the new data in low-dimensional space.
  // """
  // If we fit just a single instance then error
  private Matrix transform(Matrix X) {
    if (this.embedding_.length() == 1) {
      throw new IllegalArgumentException("Transform unavailable when model was fit with only a single data sample.");
    }
    // If we just have the original input then short circuit things
    //X = check_array(X, dtype = np.float32, accept_sparse = "csr");
    // todo caching of previous run ?
//    int x_hash = joblib.hash(X);
//    if (x_hash == this._input_hash) {
//      return this.embedding_;
//    }

    if (this._sparse_data) {
      throw new IllegalArgumentException("Transform not available for sparse input.");
    } else if (this.metric.equals("precomputed")) {
      throw new IllegalArgumentException("Transform  of new data not available for precomputed metric.");
    }

    //X = check_array(X, dtype = np.float32, order = "C");
    //random_state = check_random_state(this.transform_seed); // todo
    // final long[] rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64);
    final long[] rng_state = new long[]{rng.nextLong(), rng.nextLong(), rng.nextLong()};

    int[][] indices;
    float[][] dists;
    if (this._small_data) {
      Matrix dmat = PairwiseDistances.pairwise_distances(X, this._raw_data, /*metric = */this.metric, this._metric_kwds); // todo pairwise_distances from sklearn metrics
      //indices = np.argpartition(dmat, this._n_neighbors)[:, :this._n_neighbors];
      indices = MathUtils.subArray(MathUtils.argpartition(dmat, this._n_neighbors), this._n_neighbors);
      float[][] dmat_shortened = Utils.submatrix(dmat, indices, this._n_neighbors);
      int[][] indices_sorted = MathUtils.argsort(dmat_shortened);
      indices = Utils.submatrix(indices, indices_sorted, this._n_neighbors);
      dists = Utils.submatrix(dmat_shortened, indices_sorted, this._n_neighbors);
    } else {
      init = initialise_search(
        this._rp_forest,
        this._raw_data,
        X,
        (int) (this._n_neighbors * this.transform_queue_size),
        this._random_init,
        this._tree_init,
        rng_state);
      result = this._search(
        this._raw_data,
        this._search_graph.indptr,
        this._search_graph.indices,
        init,
        X);

      indices, dists = deheap_sort(result);
//      indices = indices[:, :this._n_neighbors];
//      dists = dists[:, :this._n_neighbors];
      indices = MathUtils.subArray(indices, this._n_neighbors);
      dists = MathUtils.subArray(dists, this._n_neighbors);
    }

    double adjusted_local_connectivity = Math.max(0, this.local_connectivity - 1.0);
    final float[][] sigmasRhos = smooth_knn_dist(dists, this._n_neighbors, /* local_connectivity=*/adjusted_local_connectivity);
    float[] sigmas = sigmasRhos[0];
    float[] rhos = sigmasRhos[1];
    final Object[] rcv = compute_membership_strengths(indices, dists, sigmas, rhos);
    final int[] rows = (int[]) rcv[0];
    final int[] cols = (int[]) rcv[1];
    final float[] vals = (float[]) rcv[2];

    CooMatrix graph = new CooMatrix(vals, rows, cols, new int[]{(int) X.length(), (int) this._raw_data.length()});

    // This was a very specially constructed graph with constant degree.
    // That lets us do fancy unpacking by reshaping the csr matrix indices
    // and data. Doing so relies on the constant degree assumption!
    CsrMatrix csr_graph = (CsrMatrix) Normalize.normalize(graph.tocsr(), "l1");
    int[][] inds = csr_graph.indices.reshape(X.length(), this._n_neighbors);
    float[][] weights = csr_graph.data.reshape(X.length(), this._n_neighbors);
    Matrix embedding = init_transform(inds, weights, this.embedding_);

    final int n_epochs;
    if (this.n_epochs == null) {
      // For smaller datasets we can use more epochs
      if (graph.length() <= 10000) {
        n_epochs = 100;
      } else {
        n_epochs = 30;
      }
    } else {
      n_epochs = this.n_epochs; // 3.0
    }

    MathUtils.zeroEntriesBelowLimit(graph.data, MathUtils.max(graph.data) / (float) n_epochs);
    graph.eliminate_zeros();

    final float[] epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs);

    int[] head = graph.row;
    int[] tail = graph.col;

    Matrix res_embedding = optimize_layout(
      embedding,
      this.embedding_.copy(), //.astype(np.float32, copy = true),  // Fixes #179 & #217
      head,
      tail,
      n_epochs,
      graph.shape()[1],
      epochs_per_sample,
      this._a,
      this._b,
      rng_state,
      this.repulsion_strength,
      this._initial_alpha,
      this.negative_sample_rate,
      this.verbose
      );

    return res_embedding;
  }
}
