/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import tagbio.umap.metric.CategoricalMetric;
import tagbio.umap.metric.EuclideanMetric;
import tagbio.umap.metric.Metric;
import tagbio.umap.metric.PrecomputedMetric;
import tagbio.umap.metric.ReducedEuclideanMetric;

// init: string (optional, default 'spectral')
//     How to initialize the low dimensional embedding. Options are:
//         * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
//         * 'random': assign initial embedding positions at random.
//         * A numpy array of initial embedding positions.

/**
 * Uniform Manifold Approximation and Projection.
 *
 * Finds a low dimensional embedding of the data that approximates an underlying manifold.
 *
 * @author Leland McInnes (Python)
 * @author Sean A. Irvine (Java port)
 * @author Richard Littin (Java port)
 */
public class Umap {

  private static final float SMOOTH_K_TOLERANCE = 1e-5F;
  private static final float MIN_K_DIST_SCALE = 1e-3F;

  private static final int SMALL_PROBLEM_THRESHOLD = 4096;

  /**
   * Compute a continuous version of the distance to the kth nearest
   * neighbor. That is, this is similar to knn-distance but allows continuous
   * k values rather than requiring an integral k. In essence we are simply
   * computing the distance such that the cardinality of fuzzy set we generate
   * is k.
   * @param distances array of shape <code>(nSamples, nNeighbors)</code>
   * Distances to nearest neighbors for each samples. Each row should be a
   * sorted list of distances to a given samples nearest neighbors.
   * @param k The number of nearest neighbors to approximate for.
   * @param nIter We need to binary search for the correct distance value. This is the
   * max number of iterations to use in such a search.
   * @param localConnectivity The local connectivity required; i.e., the number of nearest
   * neighbors that should be assumed to be connected at a local level.
   * The higher this value the more connected the manifold becomes
   * locally. In practice this should be not more than the local intrinsic
   * dimension of the manifold.
   * @param bandwidth The target bandwidth of the kernel, larger values will produce
   * larger return values.
   * @return two arrays knnDist array of shape <code>(nSamples)</code>
   * The distance to kth nearest neighbor, as suitably approximated.
   * nnDist: array of shape <code>(nSamples)</code>
   * The distance to the first nearest neighbor for each point.
   */
  private static float[][] smoothKnnDist(final float[][] distances, final float k, final int nIter, final int localConnectivity, final float bandwidth) {
    final float target = (float) (MathUtils.log2(k) * bandwidth);
    final float[] rho = new float[distances.length];
    final float[] result = new float[distances.length];

    final float meanDistances = MathUtils.mean(distances);

    for (int i = 0; i < distances.length; ++i) {
      float lo = 0;
      float hi = Float.POSITIVE_INFINITY;
      float mid = 1;

      final float[] ithDistances = distances[i];
      final float[] nonZeroDists = MathUtils.filterPositive(ithDistances);
      if (nonZeroDists.length >= localConnectivity) {
        final int index = (int) Math.floor(localConnectivity);
        final float interpolation = localConnectivity - index;
        if (index > 0) {
          rho[i] = nonZeroDists[index - 1];
          if (interpolation > SMOOTH_K_TOLERANCE) {
            rho[i] += interpolation * (nonZeroDists[index] - nonZeroDists[index - 1]);
          }
        } else {
          rho[i] = interpolation * nonZeroDists[0];
        }
      } else if (nonZeroDists.length > 0) {
        rho[i] = MathUtils.max(nonZeroDists);
      }

      for (int n = 0; n < nIter; ++n) {
        double pSum = 0.0;
        for (int j = 1; j < distances[0].length; ++j) {
          final double d = distances[i][j] - rho[i];
          pSum += d > 0 ? Math.exp(-(d / mid)) : 1;
        }

        if (Math.abs(pSum - target) < SMOOTH_K_TOLERANCE) {
          break;
        }

        if (pSum > target) {
          hi = mid;
          mid = (lo + hi) / 2.0F;
        } else {
          lo = mid;
          if (hi == Float.POSITIVE_INFINITY) {
            mid *= 2;
          } else {
            mid = (lo + hi) / 2.0F;
          }
        }
      }

      result[i] = mid;

      if (rho[i] > 0) {
        final float meanIthDistances = MathUtils.mean(ithDistances);
        if (result[i] < MIN_K_DIST_SCALE * meanIthDistances) {
          result[i] = MIN_K_DIST_SCALE * meanIthDistances;
        }
      } else {
        if (result[i] < MIN_K_DIST_SCALE * meanDistances) {
          result[i] = MIN_K_DIST_SCALE * meanDistances;
        }
      }
    }
    return new float[][]{result, rho};
  }

  static float[][] smoothKnnDist(final float[][] distances, final float k, final int localConnectivity) {
    return smoothKnnDist(distances, k, 64, localConnectivity, 1.0F);
  }

  /**
   * Compute the <code>nNeighbors</code> nearest points for each data point in <code>instances</code>
   * under <code>metric</code>. This may be exact, but more likely is approximated via
   * nearest neighbor descent.
   * @param instances The input data to compute the k-neighbor graph of.
   * @param nNeighbors The number of nearest neighbors to compute for each sample in <code>instances</code>.
   * @param metric The metric to use for the computation.
   * @param angular Whether to use angular rp trees in NN approximation.
   * @param random The random state to use for approximate NN computations.
   * @param verbose Whether to print status data during the computation.
   * @return knnIndices: array of shape <code>(nSamples, nNeighbors)</code>
   *   The indices on the <code>nNeighbors</code> closest points in the dataset.
   *   knnDists: array of shape <code>(nSamples, nNeighbors)</code>
   *   The distances to the <code>nNeighbors</code> closest points in the dataset.
   */
  static IndexedDistances nearestNeighbors(final Matrix instances, final int nNeighbors, final Metric metric, boolean angular, final Random random, final boolean verbose) {
    if (verbose) {
      Utils.message("Finding nearest neighbors");
    }

    final int[][] knnIndices;
    final float[][] knnDists;
    final List<FlatTree> rpForest;
    if (metric.equals(PrecomputedMetric.SINGLETON)) {
      // Note that this does not support sparse distance matrices yet ...
      // Compute indices of n nearest neighbors
      knnIndices = Utils.fastKnnIndices(instances, nNeighbors);
      // Compute the nearest neighbor distances
      //   (equivalent to np.sort(X)[:,:nNeighbors])
      //knnDists = X[np.arange(X.rows())[:, None],knnIndices].copy();
      knnDists = new float[knnIndices.length][nNeighbors];
      for (int i = 0; i < knnDists.length; ++i) {
        for (int j = 0; j < nNeighbors; ++j) {
          knnDists[i][j] = instances.get(i, knnIndices[i][j]);
        }
      }
      rpForest = Collections.emptyList();
    } else {
      Metric distanceFunc = metric;
      angular = distanceFunc.isAngular();

      if (instances instanceof CsrMatrix) {
        final CsrMatrix csrInstances = (CsrMatrix) instances;
        // todo this is nonsense now, since metric cannot be a string at this point
        if (Sparse.SPARSE_NAMED_DISTANCES.containsKey(metric)) {
          distanceFunc = Sparse.SPARSE_NAMED_DISTANCES.get(metric);
//          if (Sparse.sparse_need_n_features.contains(metric)) {
//            metricKwds.put("n_features", instances.cols());
//          }
        } else {
          throw new IllegalArgumentException("Metric " + metric + " not supported for sparse data");
        }
        throw new UnsupportedOperationException();
//        metric_nn_descent = Sparse.make_sparse_nn_descent(distanceFunc, tuple(metricKwds.values()));
//
//        int n_trees = 5 + (int) (Math.round(Math.pow(Y.rows(), 0.5) / 20.0));
//        int n_iters = Math.max(5, (int) (Math.round(MathUtils.log2(Y.rows()))));
//        if (verbose) {
//          Utils.message("Building RP forest with " + n_trees + " trees");
//        }
//
//        rpForest = RpTree.make_forest(X, nNeighbors, n_trees, random, angular);
//        final int[][] leaf_array = RpTree.rptree_leaf_array(rpForest);
//
//        if (verbose) {
//          Utils.message("NN descent for " + n_iters + " iterations");
//        }
//        final Object[] nn = NearestNeighborDescent.metric_nn_descent(Y.indices, Y.indptr, Y.data, Y.rows(), nNeighbors, random, /*max_candidates=*/60, /*rp_tree_init=*/true,     /*leaf_array=*/leaf_array,  /*n_iters=*/n_iters, verbose);
//        knnIndices = (int[][]) nn[0];
//        knnDists = (float[][]) nn[1];
      } else {
        final NearestNeighborDescent metricNearestNeighborsDescent = new NearestNeighborDescent(distanceFunc);
        int nTrees = 5 + (int) (Math.round(Math.pow(instances.rows(), 0.5) / 20.0));
        int nIters = Math.max(5, (int) (Math.round(MathUtils.log2(instances.rows()))));

        if (verbose) {
          Utils.message("Building random projection forest with " + nTrees + " trees");
        }
        rpForest = RandomProjectionTree.makeForest(instances, nNeighbors, nTrees, random, angular);
        final int[][] leafArray = RandomProjectionTree.rptreeLeafArray(rpForest);
        if (verbose) {
          Utils.message("NN descent for " + nIters + " iterations");
        }
        final Heap nn = metricNearestNeighborsDescent.descent(instances, nNeighbors, random, 60, true, nIters, leafArray, verbose);
        knnIndices = nn.indices();
        knnDists = nn.weights();
      }

      if (MathUtils.containsNegative(knnIndices)) {
        Utils.message("Failed to correctly find nearest neighbors for some samples. Results may be less than ideal. Try re-running with different parameters.");
      }
    }
    if (verbose) {
      Utils.message("Finished nearest neighbor search");
    }
    return new IndexedDistances(knnIndices, knnDists, rpForest);
  }

  /**
   * Construct the membership strength data for the 1-skeleton of each local
   * fuzzy simplicial set -- this is formed as a sparse matrix where each row is
   * a local fuzzy simplicial set, with a membership strength for the
   * 1-simplex to each other data point.
   * @param knnIndices array of shape <code>(nSamples, nNeighbors)</code>
   * The indices on the <code>nNeighbors</code> closest points in the dataset.
   * @param knnDists array of shape <code>(nSamples, nNeighbors)</code>
   * The distances to the <code>nNeighbors</code> closest points in the dataset.
   * @param sigmas array of shape <code>(nSamples)</code>
   * The normalization factor derived from the metric tensor approximation.
   * @param rhos array of shape <code>(nSamples)</code>
   * The local connectivity adjustment.
   * @param rowCount number or rows in the result
   * @param colCount number or columns in the result
   * @return sparse matrix of shape <code>(nSamples, nNeighbors)</code>
   */
  static CooMatrix computeMembershipStrengths(final int[][] knnIndices, final float[][] knnDists, final float[] sigmas, final float[] rhos, final int rowCount, final int colCount) {
    final int nSamples = knnIndices.length;
    final int nNeighbors = knnIndices[0].length;
    final int size = nSamples * nNeighbors;

    final int[] rows = new int[size];
    final int[] cols = new int[size];
    final float[] vals = new float[size];

    for (int i = 0; i < nSamples; ++i) {
      for (int j = 0; j < nNeighbors; ++j) {
        if (knnIndices[i][j] == -1) {
          continue;  // We didn't get the full knn for i
        }
        final float val;
        if (knnIndices[i][j] == i) {
          val = 0;
        } else if (knnDists[i][j] - rhos[i] <= 0) {
          val = 1;
        } else {
          val = (float) Math.exp(-((knnDists[i][j] - rhos[i]) / (sigmas[i])));
        }
        rows[i * nNeighbors + j] = i;
        cols[i * nNeighbors + j] = knnIndices[i][j];
        vals[i * nNeighbors + j] = val;
      }
    }
    return new CooMatrix(vals, rows, cols, rowCount, colCount);
  }

  /**
   * Given a set of data X, a neighborhood size, and a measure of distance
   * compute the fuzzy simplicial set (here represented as a fuzzy graph in
   * the form of a sparse matrix) associated to the data. This is done by
   * locally approximating geodesic distance at each point, creating a fuzzy
   * simplicial set for each such point, and then combining all the local
   * fuzzy simplicial sets into a global one via a fuzzy union.
   * @param instances The data to be modelled as a fuzzy simplicial set.
   * @param nNeighbors The number of neighbors to use to approximate geodesic distance.
   * Larger numbers induce more global estimates of the manifold that can
   * miss finer detail, while smaller values will focus on fine manifold
   * structure to the detriment of the larger picture.
   * @param random Randomness source
   * @param metric The metric to use to compute distances in high dimensional space.
   * @param knnIndices array of shape <code>(nSamples, nNeighbors)</code> or null.
   * If the k-nearest neighbors of each point has already been calculated
   * you can pass them in here to save computation time. This should be
   * an array with the indices of the k-nearest neighbors as a row for
   * each data point.
   * @param knnDists array of shape <code>(nSamples, nNeighbors)</code> or null.
   * If the k-nearest neighbors of each point has already been calculated
   * you can pass them in here to save computation time. This should be
   * an array with the distances of the k-nearest neighbors as a row for
   * each data point.
   * @param angular Whether to use angular/cosine distance for the random projection
   * forest for seeding NN-descent to determine approximate nearest
   * neighbors.
   * @param setOpMixRatio Interpolate between (fuzzy) union and intersection as the set operation
   * used to combine local fuzzy simplicial sets to obtain a global fuzzy
   * simplicial sets. Both fuzzy set operations use the product t-norm.
   * The value of this parameter should be between 0.0 and 1.0; a value of
   * 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
   * intersection.
   * @param localConnectivity The local connectivity required; i.e., the number of nearest
   * neighbors that should be assumed to be connected at a local level.
   * The higher this value the more connected the manifold becomes
   * locally. In practice this should be not more than the local intrinsic
   * dimension of the manifold.
   * @param verbose Whether to report information on the current progress of the algorithm.
   * @return A fuzzy simplicial set represented as a sparse matrix. The <code>(i, j)</code>
   * entry of the matrix represents the membership strength of the
   * 1-simplex between the ith and jth sample points.
   */
  static Matrix fuzzySimplicialSet(final Matrix instances, final int nNeighbors, final Random random, final Metric metric, int[][] knnIndices, float[][] knnDists, final boolean angular, final float setOpMixRatio, final int localConnectivity, final boolean verbose) {

    if (knnIndices == null || knnDists == null) {
      final IndexedDistances nn = nearestNeighbors(instances, nNeighbors, metric, angular, random, verbose);
      knnIndices = nn.getIndices();
      knnDists = nn.getDistances();
    }

    final float[][] sigmasRhos = smoothKnnDist(knnDists, nNeighbors, localConnectivity);
    final float[] sigmas = sigmasRhos[0];
    final float[] rhos = sigmasRhos[1];

    final Matrix result = computeMembershipStrengths(knnIndices, knnDists, sigmas, rhos, instances.rows(), instances.rows()).eliminateZeros();
    final Matrix prodMatrix = result.hadamardMultiplyTranspose();

    return result.addTranspose().subtract(prodMatrix).multiply(setOpMixRatio).add(prodMatrix.multiply(1.0F - setOpMixRatio)).eliminateZeros();
  }

  /**
   * Reset the local connectivity requirement -- each data sample should
   * have complete confidence in at least one 1-simplex in the simplicial set.
   * We can enforce this by locally rescaling confidences, and then remerging the
   * different local simplicial sets together.
   * @param simplicialSet The simplicial set for which to recalculate with respect to local connectivity.
   * @return The recalculated simplicial set, now with the local connectivity assumption restored.
   */
  private static Matrix resetLocalConnectivity(final Matrix simplicialSet) {
    final Matrix nss = simplicialSet.rowNormalize();
    final Matrix prodMatrix = nss.hadamardMultiplyTranspose();
    return nss.addTranspose().subtract(prodMatrix).eliminateZeros();
  }

  /**
   * Combine a fuzzy simplicial set with another fuzzy simplicial set
   * generated from categorical data using categorical distances. The target
   * data is assumed to be categorical label data (a vector of labels),
   * and this will update the fuzzy simplicial set to respect that label data.
   * @param simplicialSet The input fuzzy simplicial set.
   * @param target The categorical labels to use in the intersection.
   * @param unknownDist The distance an unknown label (-1) is assumed to be from any point.
   * @param farDist The distance between unmatched labels.
   * @return The resulting intersected fuzzy simplicial set.
   */
  private static Matrix categoricalSimplicialSetIntersection(final CooMatrix simplicialSet, final float[] target, final float unknownDist, final float farDist) {
    simplicialSet.fastIntersection(target, unknownDist, farDist);
    return resetLocalConnectivity(simplicialSet.eliminateZeros());
  }

  private static Matrix categoricalSimplicialSetIntersection(final CooMatrix simplicialSet, final float[] target, final float farDist) {
    return categoricalSimplicialSetIntersection(simplicialSet, target, 1.0F, farDist);
  }

  private static Matrix generalSimplicialSetIntersection(final Matrix simplicialSet1, final Matrix simplicialSet2, final float weight) {

    final CooMatrix result = simplicialSet1.add(simplicialSet2).toCoo();
    final CsrMatrix left = simplicialSet1.toCsr();
    final CsrMatrix right = simplicialSet2.toCsr();

    Sparse.generalSsetIntersection(left.indptr(), left.indicies(), left.data(), right.indptr(), right.indicies(), right.data(), result.row(), result.col(), result.data(), weight);

    return result;
  }

  /**
   * Given a set of weights and number of epochs generate the number of
   * epochs per sample for each weight.
   * @param weights The weights of how much we wish to sample each 1-simplex.
   * @param nEpochs The total number of epochs we want to train for.
   * @return An array of number of epochs per sample, one for each 1-simplex.
   */
  static float[] makeEpochsPerSample(final float[] weights, final int nEpochs) {
    final float[] result = new float[weights.length];
    Arrays.fill(result, -1.0F);
    final float[] nSamples = MathUtils.multiply(MathUtils.divide(weights, MathUtils.max(weights)), nEpochs);
    for (int k = 0; k < nSamples.length; ++k) {
      if (nSamples[k] > 0) {
        result[k] = (float) nEpochs / nSamples[k];
      }
    }
    return result;
  }

  /**
   * Standard clamping of a value into a fixed range (in this case -4.0 to 4.0)
   * @param val The value to be clamped.
   * @return clamped value
   */
  static float clip(final float val) {
    return val > 4 ? 4 : val < -4 ? -4 : val;
  }

  /**
   * Improve an embedding using stochastic gradient descent to minimize the
   * fuzzy set cross entropy between the 1-skeletons of the high dimensional
   * and low dimensional fuzzy simplicial sets. In practice this is done by
   * sampling edges based on their membership strength (with the (1-p) terms
   * coming from negative sampling similar to word2vec).
   * @param headEmbedding array of shape <code>(nSamples, nComponents)</code>
   * The initial embedding to be improved by SGD.
   * @param tailEmbedding array of shape <code>(sourceSamples, nComponents)</code>
   * The reference embedding of embedded points. If not embedding new
   * previously unseen points with respect to an existing embedding this
   * is simply the <code>headEmbedding</code> (again); otherwise it provides the
   * existing embedding to embed with respect to.
   * @param head array of shape (n_1_simplices)
   * The indices of the heads of 1-simplices with non-zero membership.
   * @param tail array of shape (n_1_simplices)
   * The indices of the tails of 1-simplices with non-zero membership.
   * @param nEpochs The number of training epochs to use in optimization.
   * @param nVertices The number of vertices (0-simplices) in the dataset.
   * @param epochsPerSample A float value of the number of epochs per 1-simplex. 1-simplices with
   * weaker membership strength will have more epochs between being sampled.
   * @param a Parameter of differentiable approximation of right adjoint functor
   * @param b Parameter of differentiable approximation of right adjoint functor
   * @param random Random source
   * @param gamma Weight to apply to negative samples.
   * @param initialAlpha Initial learning rate for the SGD.
   * @param negativeSampleRate Number of negative samples to use per positive sample.
   * @param verbose Whether to report information on the current progress of the algorithm.
   * @return array of shape <code>(nSamples, nComponents)</code> The optimized embedding.
   */
  private static Matrix optimizeLayout(final Matrix headEmbedding, final Matrix tailEmbedding, final int[] head, final int[] tail, final int nEpochs, final int nVertices, final float[] epochsPerSample, final float a, final float b, final Random random, final float gamma, final float initialAlpha, final float negativeSampleRate, final boolean verbose) {

    assert headEmbedding instanceof DefaultMatrix; // because this routine directly modifies contents of rows in the matrix

    final int dim = headEmbedding.cols();
    final boolean moveOther = headEmbedding.rows() == tailEmbedding.rows();
    float alpha = initialAlpha;

    final float[] epochsPerNegativeSample = MathUtils.divide(epochsPerSample, negativeSampleRate);
    final float[] epochOfNextNegativeSample = Arrays.copyOf(epochsPerNegativeSample, epochsPerNegativeSample.length);
    final float[] epochOfNextSample = Arrays.copyOf(epochsPerSample, epochsPerSample.length);

    for (int n = 0; n < nEpochs; ++n) {
      for (int i = 0; i < epochsPerSample.length; ++i) {
        if (epochOfNextSample[i] <= n) {
          final int j = head[i];
          final int k = tail[i];
          // todo this assumes that "current" is a pointer to the internal matrix data
          final float[] current = headEmbedding.row(j);
          float[] other = tailEmbedding.row(k);

          float distSquared = ReducedEuclideanMetric.SINGLETON.distance(current, other);

          float gradCoeff;
          if (distSquared > 0.0) {
            gradCoeff = (float) ((-2.0 * a * b * Math.pow(distSquared, b - 1.0)) / (a * Math.pow(distSquared, b) + 1.0));
          } else {
            gradCoeff = 0;
          }

          for (int d = 0; d < dim; ++d) {
            final float gradD = clip(gradCoeff * (current[d] - other[d]));
            current[d] += gradD * alpha;
            if (moveOther) {
              other[d] += -gradD * alpha;
            }
          }

          epochOfNextSample[i] += epochsPerSample[i];

          final int nNegSamples = (int) ((n - epochOfNextNegativeSample[i]) / epochsPerNegativeSample[i]);

          for (int p = 0; p < nNegSamples; ++p) {
            final int kr = random.nextInt(nVertices);
            other = tailEmbedding.row(kr);
            distSquared = ReducedEuclideanMetric.SINGLETON.distance(current, other);

            if (distSquared > 0) {
              gradCoeff = 2.0F * gamma * b / (float) ((0.001 + distSquared) * (a * Math.pow(distSquared, b) + 1));
            } else if (j == kr) {
              continue;
            } else {
              gradCoeff = 0;
            }

            for (int d = 0; d < dim; ++d) {
              final float gradD = gradCoeff > 0.0 ? clip(gradCoeff * (current[d] - other[d])) : 4;
              current[d] += gradD * alpha;
            }
          }

          epochOfNextNegativeSample[i] += nNegSamples * epochsPerNegativeSample[i];
        }
      }

      alpha = initialAlpha * (1 - (float) n / (float) (nEpochs));

      if (verbose && n % (nEpochs / 10) == 0) {
        Utils.message("Completed " + n + "/" + nEpochs);
      }
    }
    return headEmbedding;
  }

  /**
   * Perform a fuzzy simplicial set embedding, using a specified
   * initialisation method and then minimizing the fuzzy set cross entropy
   * between the 1-skeletons of the high and low dimensional fuzzy simplicial
   *  sets.
   * @param data array of shape <code>(nSamples, nFeatures)</code>
   * The source data to be embedded by UMAP.
   * @param graphIn The 1-skeleton of the high dimensional fuzzy simplicial set as
   * represented by a graph for which we require a sparse matrix for the
   * (weighted) adjacency matrix.
   * @param nComponents The dimensionality of the euclidean space into which to embed the data.
   * @param initialAlpha Initial learning rate for the SGD.
   * @param a Parameter of differentiable approximation of right adjoint functor.
   * @param b Parameter of differentiable approximation of right adjoint functor.
   * @param gamma Weight to apply to negative samples.
   * @param negativeSampleRate The number of negative samples to select per positive sample
   * in the optimization process. Increasing this value will result
   * in greater repulsive force being applied, greater optimization
   * cost, but slightly more accuracy.
   * @param nEpochs The number of training epochs to be used in optimizing the
   * low dimensional embedding. Larger values result in more accurate
   * embeddings. If 0 is specified a value will be selected based on
   * the size of the input dataset (200 for large datasets, 500 for small).
   * @param init How to initialize the low dimensional embedding. Options are:
   * * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
   * * 'random': assign initial embedding positions at random.
   * * A numpy array of initial embedding positions.
   * @param random random source
   * @param metric The metric used to measure distance in high dimensional space; used if
   * multiple connected components need to be layed out.
   * @param verbose Whether to report information on the current progress of the algorithm.
   * @return array of shape <code>(nSamples, nComponents)</code>
   * The optimized of <code>graph</code> into an <code>nComponents</code> dimensional
   * Euclidean space.
   */
  private static Matrix simplicialSetEmbedding(Matrix data, Matrix graphIn, int nComponents, float initialAlpha, float a, float b, float gamma, int negativeSampleRate, int nEpochs, String init, Random random, Metric metric, boolean verbose) {

    CooMatrix graph = graphIn.toCoo();
    int nVertices = graph.cols();

    if (nEpochs <= 0) {
      // For smaller datasets we can use more epochs
      if (graph.rows() <= 10000) {
        nEpochs = 500;
      } else {
        nEpochs = 200;
      }
    }

    final float[] graphData = graph.data();
    MathUtils.zeroEntriesBelowLimit(graphData, MathUtils.max(graphData) / (float) nEpochs);
    graph = (CooMatrix) graph.eliminateZeros();

    Matrix embedding;
    if ("random".equals(init)) {
      //embedding = random.uniform(low = -10.0, high = 10.0, size = (graph.rows(), nComponents)).astype(np.float32);
      embedding = new DefaultMatrix(MathUtils.uniform(random, -10, 10, graph.rows(), nComponents));
    } else if ("spectral".equals(init)) {
      throw new UnsupportedOperationException();
//      // We add a little noise to avoid local minima for optimization to come
//      float[][] initialisation = Spectral.spectral_layout(data, graph, nComponents, random, /*metric=*/metric, /*metric_kwds=*/metric_kwds);
//      float expansion = 10.0 / Math.abs(initialisation).max();
//      embedding = (MathUtils.multiply(initialisation, expansion)).astype(np.float32) + random.normal(scale = 0.0001, size =[graph.rows(), nComponents]).astype(np.float32);
    } else {
      // Situation where init contains prepared data
      throw new UnsupportedOperationException();
//      init_data = np.array(init);
//      if (len(init_data.shape) == 2) {
//        if (np.unique(init_data, /*axis =*/ 0).length < init_data.length) {
//          tree = KDTree(init_data);
//          float[][] dist /*, ind*/ = tree.query(init_data, k = 2);
//          double nndist = MathUtils.mean(dist, 1);
//          embedding = init_data + random.normal(scale = 0.001 * nndist, size = init_data.shape).astype(np.float32);
//        } else {
//          embedding = init_data;
//        }
//      }
    }

    final float[] epochsPerSample = makeEpochsPerSample(graph.data(), nEpochs);
    final int[] head = graph.row();
    final int[] tail = graph.col();

    // so (head, tail, epochsPerSample) is like a CooMatrix

    return optimizeLayout(embedding, embedding, head, tail, nEpochs, nVertices, epochsPerSample, a, b, random, gamma, initialAlpha, negativeSampleRate, verbose);
  }

  /**
   * Given indices and weights and an original embeddings
   * initialize the positions of new points relative to the
   * indices and weights (of their neighbors in the source data).
   * @param indices array of shape <code>(nNewSamples, nNeighbors)</code>
   * The indices of the neighbors of each new sample
   * @param weights array of shape <code>(nNewSamples, nNeighbors)</code>
   * The membership strengths of associated 1-simplices
   * for each of the new samples.
   * @param embedding array of shape <code>(nSamples, dim)</code>
   * The original embedding of the source data.
   * @return array of shape <code>(nNewSamples, dim)</code>
   * An initial embedding of the new sample points.
   */
  private static Matrix initTransform(final int[][] indices, final float[][] weights, final Matrix embedding) {
    final float[][] result = new float[indices.length][embedding.cols()];
    for (int i = 0; i < indices.length; ++i) {
      for (int j = 0; j < indices[i].length; ++j) {
        for (int d = 0; d < embedding.cols(); ++d) {
          result[i][d] += (weights[i][j] * embedding.get(indices[i][j], d));
        }
      }
    }
    return new DefaultMatrix(result);
  }

//  private static double curve(final float x, final float a, final float b) {
//    return 1.0 / (1.0 + a * Math.pow(x, 2 * b));
//

  private static float[][] mSpreadDistAs = {
    {},
    {},
    {},
    {},
    {},
    {0.00000F, 5.45377F, 5.06931F, 4.63694F, 4.17619F, 3.70629F, 3.24352F, 2.80060F, 2.38675F, 2.00802F, 1.66772F, 1.36696F},
    {0.00000F, 4.00306F, 3.65757F, 3.30358F, 2.95032F, 2.60637F, 2.27854F, 1.97183F, 1.68955F, 1.43356F, 1.20450F, 1.00209F, 0.82529F, 0.67254F},
    {0.00000F, 3.10329F, 2.81124F, 2.52444F, 2.24750F, 1.98435F, 1.73792F, 1.51015F, 1.30214F, 1.11425F, 0.94625F, 0.79748F, 0.66690F, 0.55330F, 0.45530F, 0.37144F},
    {0.00000F, 2.49880F, 2.25458F, 2.02012F, 1.79779F, 1.58943F, 1.39624F, 1.21887F, 1.05750F, 0.91192F, 0.78165F, 0.66593F, 0.56388F, 0.47451F, 0.39679F, 0.32963F, 0.27198F, 0.22280F},
    {0.00000F, 2.06902F, 1.86409F, 1.66991F, 1.48771F, 1.31831F, 1.16212F, 1.01922F, 0.88942F, 0.77230F, 0.66732F, 0.57375F, 0.49090F, 0.41788F, 0.35395F, 0.29823F, 0.24996F, 0.20833F, 0.17265F, 0.14221F},
    {0.00000F, 1.75022F, 1.57694F, 1.41400F, 1.26206F, 1.12144F, 0.99218F, 0.87410F, 0.76687F, 0.67003F, 0.58303F, 0.50526F, 0.43607F, 0.37481F, 0.32082F, 0.27345F, 0.23206F, 0.19607F, 0.16490F, 0.13804F, 0.11498F, 0.09527F},
    {0.00000F, 1.50587F, 1.35804F, 1.21967F, 1.09109F, 0.97240F, 0.86343F, 0.76395F, 0.67354F, 0.59177F, 0.51814F, 0.45211F, 0.39314F, 0.34068F, 0.29420F, 0.25316F, 0.21708F, 0.18544F, 0.15784F, 0.13382F, 0.11302F, 0.09505F, 0.07960F, 0.06635F},
    {0.00000F, 1.31360F, 1.18637F, 1.06759F, 0.95743F, 0.85584F, 0.76262F, 0.67747F, 0.60002F, 0.52985F, 0.46650F, 0.40952F, 0.35843F, 0.31280F, 0.27218F, 0.23613F, 0.20424F, 0.17611F, 0.15137F, 0.12970F, 0.11076F, 0.09428F, 0.07997F, 0.06758F, 0.05690F, 0.04771F},
    {0.00000F, 1.15902F, 1.04861F, 0.94567F, 0.85027F, 0.76232F, 0.68159F, 0.60780F, 0.54058F, 0.47955F, 0.42432F, 0.37449F, 0.32967F, 0.28950F, 0.25358F, 0.22154F, 0.19304F, 0.16777F, 0.14542F, 0.12571F, 0.10837F, 0.09316F, 0.07985F, 0.06824F, 0.05813F, 0.04936F, 0.04178F, 0.03524F},
    {0.00000F, 1.03253F, 0.93595F, 0.84598F, 0.76261F, 0.68572F, 0.61510F, 0.55045F, 0.49148F, 0.43780F, 0.38915F, 0.34512F, 0.30540F, 0.26965F, 0.23756F, 0.20883F, 0.18315F, 0.16028F, 0.13992F, 0.12189F, 0.10591F, 0.09182F, 0.07940F, 0.06848F, 0.05892F, 0.05055F, 0.04325F, 0.03690F, 0.03139F, 0.02662F},
    {0.00000F, 0.92742F, 0.84237F, 0.76312F, 0.68968F, 0.62189F, 0.55955F, 0.50242F, 0.45021F, 0.40260F, 0.35933F, 0.32008F, 0.28455F, 0.25248F, 0.22359F, 0.19762F, 0.17432F, 0.15347F, 0.13483F, 0.11822F, 0.10345F, 0.09033F, 0.07870F, 0.06843F, 0.05936F, 0.05137F, 0.04437F, 0.03821F, 0.03283F, 0.02814F, 0.02405F, 0.02050F},
    {0.00000F, 0.83896F, 0.76357F, 0.69330F, 0.62813F, 0.56793F, 0.51250F, 0.46161F, 0.41501F, 0.37246F, 0.33368F, 0.29843F, 0.26643F, 0.23745F, 0.21126F, 0.18763F, 0.16635F, 0.14723F, 0.13008F, 0.11472F, 0.10100F, 0.08875F, 0.07784F, 0.06815F, 0.05954F, 0.05192F, 0.04518F, 0.03924F, 0.03401F, 0.02941F, 0.02537F, 0.02184F, 0.01875F, 0.01606F},
  };
  private static float[][] mSpreadDistBs = {
    {},
    {},
    {},
    {},
    {},
    {0.00000F, 0.89506F, 1.00301F, 1.11225F, 1.22256F, 1.33417F, 1.44746F, 1.56295F, 1.68123F, 1.80304F, 1.92924F, 2.06090F},
    {0.00000F, 0.87728F, 0.96682F, 1.05750F, 1.14889F, 1.24105F, 1.33417F, 1.42845F, 1.52416F, 1.62170F, 1.72144F, 1.82373F, 1.92923F, 2.03858F},
    {0.00000F, 0.86464F, 0.94109F, 1.01853F, 1.09656F, 1.17513F, 1.25429F, 1.33417F, 1.41491F, 1.49668F, 1.57967F, 1.66414F, 1.75035F, 1.83859F, 1.92923F, 2.02267F},
    {0.00000F, 0.85520F, 0.92186F, 0.98941F, 1.05750F, 1.12598F, 1.19486F, 1.26423F, 1.33417F, 1.40477F, 1.47610F, 1.54836F, 1.62170F, 1.69629F, 1.77222F, 1.84979F, 1.92924F, 2.01085F},
    {0.00000F, 0.84788F, 0.90695F, 0.96682F, 1.02719F, 1.08787F, 1.14889F, 1.21024F, 1.27197F, 1.33417F, 1.39686F, 1.46019F, 1.52416F, 1.58901F, 1.65466F, 1.72144F, 1.78930F, 1.85855F, 1.92924F, 2.00165F},
    {0.00000F, 0.84206F, 0.89506F, 0.94882F, 1.00301F, 1.05750F, 1.11225F, 1.16727F, 1.22256F, 1.27818F, 1.33417F, 1.39057F, 1.44746F, 1.50490F, 1.56295F, 1.62170F, 1.68123F, 1.74165F, 1.80304F, 1.86553F, 1.92924F, 1.99431F},
    {0.00000F, 0.83729F, 0.88535F, 0.93409F, 0.98326F, 1.03268F, 1.08235F, 1.13221F, 1.18231F, 1.23264F, 1.28324F, 1.33417F, 1.38541F, 1.43709F, 1.48916F, 1.54179F, 1.59489F, 1.64870F, 1.70309F, 1.75832F, 1.81431F, 1.87129F, 1.92923F, 1.98835F},
    {0.00000F, 0.83333F, 0.87728F, 0.92186F, 0.96682F, 1.01206F, 1.05750F, 1.10311F, 1.14889F, 1.19486F, 1.24105F, 1.28747F, 1.33417F, 1.38116F, 1.42845F, 1.47610F, 1.52416F, 1.57268F, 1.62170F, 1.67128F, 1.72144F, 1.77222F, 1.82373F, 1.87604F, 1.92923F, 1.98338F},
    {0.00000F, 0.82999F, 0.87046F, 0.91153F, 0.95296F, 0.99464F, 1.03652F, 1.07852F, 1.12068F, 1.16301F, 1.20550F, 1.24819F, 1.29109F, 1.33417F, 1.37750F, 1.42113F, 1.46506F, 1.50934F, 1.55401F, 1.59903F, 1.64450F, 1.69046F, 1.73696F, 1.78405F, 1.83178F, 1.88014F, 1.92924F, 1.97915F},
    {0.00000F, 0.82713F, 0.86464F, 0.90269F, 0.94109F, 0.97973F, 1.01853F, 1.05750F, 1.09656F, 1.13581F, 1.17513F, 1.21464F, 1.25429F, 1.29413F, 1.33417F, 1.37440F, 1.41491F, 1.45562F, 1.49668F, 1.53798F, 1.57967F, 1.62170F, 1.66414F, 1.70703F, 1.75035F, 1.79425F, 1.83859F, 1.88363F, 1.92924F, 1.97558F},
    {0.00000F, 0.82465F, 0.85960F, 0.89506F, 0.93083F, 0.96682F, 1.00301F, 1.03929F, 1.07570F, 1.11225F, 1.14889F, 1.18565F, 1.22256F, 1.25960F, 1.29678F, 1.33417F, 1.37172F, 1.40946F, 1.44746F, 1.48570F, 1.52416F, 1.56295F, 1.60206F, 1.64145F, 1.68124F, 1.72144F, 1.76198F, 1.80304F, 1.84462F, 1.88662F, 1.92923F, 1.97251F},
    {0.00000F, 0.82249F, 0.85520F, 0.88837F, 0.92186F, 0.95555F, 0.98941F, 1.02340F, 1.05750F, 1.09170F, 1.12598F, 1.16036F, 1.19486F, 1.22948F, 1.26423F, 1.29912F, 1.33417F, 1.36938F, 1.40477F, 1.44033F, 1.47610F, 1.51211F, 1.54836F, 1.58489F, 1.62170F, 1.65883F, 1.69629F, 1.73407F, 1.77222F, 1.81079F, 1.84979F, 1.88926F, 1.92923F, 1.96975F},
  };

  private static float findValue(float[][] spreadDist, int spreadIndex, int distIndex, float spreadDelta, float distDelta) {
    float start = spreadDist[spreadIndex][distIndex] + distDelta * (spreadDist[spreadIndex][distIndex+1] - spreadDist[spreadIndex][distIndex]);
    float end = spreadDist[spreadIndex+1][distIndex] + distDelta * (spreadDist[spreadIndex+1][distIndex+1] - spreadDist[spreadIndex+1][distIndex]);
    float val = start + spreadDelta * (end - start);
    //System.out.println(spreadDelta + " : " + distDelta + " : " + start + " : " + end + " : " + val);
    return val;
  }

  // look up table base curve fitting
  // averages values for locations between known spread/minDist pairs
  private static float[] curveFit(final float spread, final float minDist) {
    if (spread < 0.5F || spread > 1.5F) {
      throw new IllegalArgumentException("Spread must be in the range 0.5 < spread <= 1.5, got : " + spread);
    }
    if (minDist < 0 || minDist > spread) {
      throw new IllegalArgumentException("Expecting 0 < minDist < " + spread + ", got : " + minDist);
    }
    int spreadIndex = (int) (10 * spread);
    float spreadDelta = (10 * spread - spreadIndex) / 10.0F;
    int distIndex = (int) (20 * minDist);
    float distDelta = (20 * minDist - distIndex) / 20.0F;
    float a = findValue(mSpreadDistAs, spreadIndex, distIndex, spreadDelta, distDelta);
    float b = findValue(mSpreadDistBs, spreadIndex, distIndex, spreadDelta, distDelta);
    return new float[] {a, b};
  }

  // Fit a, b params for the differentiable curve used in lower
  // dimensional fuzzy simplicial complex construction. We want the
  // smooth curve (from a pre-defined family with simple gradient) that
  // best matches an offset exponential decay.
  private static float[] findAbParams(float spread, float minDist) {
    //System.out.println("find_ab_params(" + spread + ", " + minDist + ")");
    /*
    float[] xv = MathUtils.linspace(0, spread * 3, 300);
    float[] yv = new float[xv.length];
    //  yv[xv < minDist] = 1.0;
    //  yv[xv >= minDist] = Math.exp(-(xv[xv >= minDist] - minDist) / spread   );
    for (int k = 0; k < yv.length; ++k) {
      if (xv[k] < minDist) {
        yv[k] = 1.0F;
      } else {
        yv[k] = (float) Math.exp(-(xv[k] - minDist) / spread);
      }
    }

    final float[] params = Curve.curve_fit(xv, yv); // todo curve_fit in scipy
    return new float[]{params[0], params[1]};
    */
    return curveFit(spread, minDist);
/*)
    if (spread == 1.0F && minDist == 0.1F) {
      return new float[]{1.57694346F, 0.89506088F};
    }
    throw new UnsupportedOperationException();

 */
  }

  private boolean mAngularRpForest = false;
  private String mInit = "spectral";
  private int mNNeighbors = 15;
  private int mNComponents = 2;
  private Integer mNEpochs = null;
  private Metric mMetric = EuclideanMetric.SINGLETON;
  private float mLearningRate = 1.0F;
  private float mRepulsionStrength = 1.0F;
  private float mMinDist = 0.1F;
  private float mSpread = 1.0F;
  private float mSetOpMixRatio = 1.0F;
  private int mLocalConnectivity = 1;
  private int mNegativeSampleRate = 5;
  private float mTransformQueueSize = 4.0F;
  private Metric mTargetMetric = CategoricalMetric.SINGLETON;
  private int mTargetNNeighbors = -1;
  private float mTargetWeight = 0.5F;
  private int mTransformSeed = 42;
  private boolean mVerbose = false;
  private Float mA = null;
  private Float mB = null;
  private Random mRandom = new Random(42);

  private float mInitialAlpha;
  private int mRunNNeighbors;
  private boolean mSparseData;
  private float mRunA;
  private float mRunB;
  private Matrix mRawData;
  private SearchGraph mSearchGraph = null;
  private int[][] mKnnIndices;
  private float[][] mKnnDists;
  private List<FlatTree> mRpForest;
  private boolean mSmallData;
  private Metric mDistanceFunc;
  private Matrix mGraph;
  private Matrix mEmbedding;
  private NearestNeighborSearch mSearch;
  private NearestNeighborRandomInit mRandomInit;
  private NearestNeighborTreeInit mTreeInit;

  public void setInit(final String init) {
    mInit = init;
  }

  /**
   * Set the size local neighborhood (in terms of number of neighboring
   * sample points) used for manifold approximation. Larger values
   * result in more global views of the manifold, while smaller
   * values result in more local data being preserved. In general
   * values should be in the range 2 to 100.  The default is 15.
   * @param neighbors number of neighbors
   */
  public void setNumberNearestNeighbours(final int neighbors) {
    if (neighbors < 2) {
      throw new IllegalArgumentException("Number of neighbors must be greater than 2.");
    }
    mNNeighbors = neighbors;
  }

  /**
   * Set the dimension of the space to embed into. This defaults to 2 to
   * provide easy visualization, but can reasonably be set to any
   * integer value in the range 2 to 100.
   * @param components dimension of embedding space
   */
  public void setNumberComponents(final int components) {
    if (components < 1) {
      throw new IllegalArgumentException("Number of components must be greater than 0.");
    }
    mNComponents = components;
  }

  /**
   * Set the number of training epochs to be used in optimizing the
   * low dimensional embedding. Larger values result in more accurate
   * embeddings. If null is specified a value will be selected based on
   * the size of the input dataset (200 for large datasets, 500 for small).
   * The minimum value is 11.
   * @param epochs number of epochs or null
   */
  public void setNumberEpochs(final Integer epochs) {
    if (epochs != null && epochs <= 10) {
      throw new IllegalArgumentException("Epochs must be larger than 10.");
    }
    mNEpochs = epochs;
  }

  /**
   * Set the metric to use to compute distances in high dimensional space.  If the
   * metric requires additional parameters, then they are assumed to have been
   * already appropriately initialized.
   * @param metric metric function
   */
  public void setMetric(final Metric metric) {
    if (metric == null) {
      throw new NullPointerException("Null metric not permitted.");
    }
    mMetric = metric;
  }

  /**
   * Set the metric to use to compute distances in high dimensional space by name.
   * Valid string metrics include:
   * euclidean,
   * manhattan,
   * chebyshev,
   * minkowski,
   * canberra,
   * braycurtis,
   * cosine,
   * correlation,
   * haversine,
   * hamming,
   * jaccard,
   * dice,
   * russelrao,
   * kulsinski,
   * rogerstanimoto,
   * sokalmichener,
   * sokalsneath,
   * yule.
   * @param metric metric function specified by name
   */
  public void setMetric(final String metric) {
    setMetric(Metric.getMetric(metric));
  }

  /**
   * Set the initial learning rate for the embedding optimization.
   * Default 1.0.
   * @param rate learning rate
   */
  public void setLearningRate(final float rate) {
    if (rate <= 0.0) {
      throw new IllegalArgumentException("Learning rate must be positive.");
    }
    mLearningRate = rate;
  }

  /**
   * Set weighting applied to negative samples in low dimensional embedding
   * optimization. Values higher than one will result in greater weight
   * being given to negative samples. Default 1.0.
   * @param repulsionStrength repulsion strength
   */
  public void setRepulsionStrength(final float repulsionStrength) {
    if (repulsionStrength < 0.0) {
      throw new IllegalArgumentException("Repulsion strength cannot be negative.");
    }
    mRepulsionStrength = repulsionStrength;
  }

  /**
   * Set the effective minimum distance between embedded points. Smaller values
   * will result in a more clustered/clumped embedding where nearby points
   * on the manifold are drawn closer together, while larger values will
   * result on a more even dispersal of points. The value should be set
   * relative to the <code>spread</code> value, which determines the scale at which
   * embedded points will be spread out. Default 0.1.
   * @param minDist minimum distance
   */
  public void setMinDist(final float minDist) {
    if (minDist < 0.0) {
      throw new IllegalArgumentException("Minimum distance must be greater than 0.0.");
    }
    mMinDist = minDist;
  }

  /**
   * Set the effective scale of embedded points. In combination with <code>minDist</code>
   * this determines how clustered/clumped the embedded points are. Default 1.0.
   * @param spread spread value
   */
  public void setSpread(final float spread) {
    mSpread = spread;
  }

  /**
   * Interpolate between (fuzzy) union and intersection as the set operation
   * used to combine local fuzzy simplicial sets to obtain a global fuzzy
   * simplicial sets. Both fuzzy set operations use the product t-norm.
   * The value of this parameter should be between 0.0 and 1.0; a value of
   * 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
   * intersection. Default 1.0.
   * @param setOpMixRatio set operation mixing ratio
   */
  public void setSetOpMixRatio(final float setOpMixRatio) {
    if (setOpMixRatio < 0.0 || setOpMixRatio > 1.0) {
      throw new IllegalArgumentException("Set operation mixing ratio be between 0.0 and 1.0.");
    }
    mSetOpMixRatio = setOpMixRatio;
  }

  /**
   * Set the local connectivity required; i.e., the number of nearest
   * neighbors that should be assumed to be connected at a local level.
   * The higher this value the more connected the manifold becomes
   * locally. In practice this should be not more than the local intrinsic
   * dimension of the manifold. Default 1.
   * @param localConnectivity local connectivity
   */
  public void setLocalConnectivity(final int localConnectivity) {
    mLocalConnectivity = localConnectivity;
  }

  /**
   * Set the number of negative samples to select per positive sample
   * in the optimization process. Increasing this value will result
   * in greater repulsive force being applied, greater optimization
   * cost, but slightly more accuracy. Default 5.
   * @param negativeSampleRate negative sample rate
   */
  public void setNegativeSampleRate(final int negativeSampleRate) {
    if (negativeSampleRate <= 0) {
      throw new IllegalArgumentException("Negative sample rate must be positive.");
    }
    mNegativeSampleRate = negativeSampleRate;
  }

  /**
   * Set the metric used to measure distance for a target array is using supervised
   * dimension reduction. By default this is <code>CategoricalMetric.SINGLETON</code> which will measure
   * distance in terms of whether categories match or are different. Furthermore,
   * if semi-supervised is required target values of -1 will be treated as
   * unlabelled under the <code>CategoricalMetric</code> metric. If the target array takes
   * continuous values (e.g. for a regression problem) then metric of 'l1'
   * or 'l2' is probably more appropriate.
   * @param targetMetric target metric
   */
  public void setTargetMetric(final Metric targetMetric) {
    mTargetMetric = targetMetric;
  }

  /**
   * Set the target metric by name (see <code>setMetric</code> for a list of values).
   * @param targetMetric target metric
   */
  public void setTargetMetric(final String targetMetric) {
    setTargetMetric(Metric.getMetric(targetMetric));
  }

  /**
   * If true, turn on additional diagnostic output.
   * @param verbose verbose level
   */
  public void setVerbose(final boolean verbose) {
    mVerbose = verbose;
  }

  /**
   * Set the random number generator to be used.
   * @param random randomness source
   */
  public void setRandom(final Random random) {
    mRandom = random;
  }

  /**
   * Set the seed of the random number generator.
   * @param seed seed value
   */
  public void setSeed(final long seed) {
    mRandom.setSeed(seed);
  }

  /**
   * For transform operations (embedding new points using a trained model
   * this will control how aggressively to search for nearest neighbors.
   * Larger values will result in slower performance but more accurate
   * nearest neighbor evaluation. Default 4.0.
   * @param transformQueueSize
   */
  public void setTransformQueueSize(final float transformQueueSize) {
    mTransformQueueSize = transformQueueSize;
  }

  /**
   * Whether to use an angular random projection forest to initialise
   * the approximate nearest neighbor search. This can be faster, but is
   * mostly on useful for metric that use an angular style distance such
   * as cosine, correlation etc. In the case of those metrics angular forests
   * will be chosen automatically.
   * @param angularRpForest true for an angular random projection forest
   */
  public void setAngularRpForest(final boolean angularRpForest) {
    mAngularRpForest = angularRpForest;
  }

  /**
   * The number of nearest neighbors to use to construct the target simplicial
   * set. If set to -1 use the <code>nNeighbors</code> value.
   * @param targetNNeighbors target nearest neighbours
   */
  public void setTargetNNeighbors(final int targetNNeighbors) {
    if (targetNNeighbors < 2 && targetNNeighbors != -1) {
      throw new IllegalArgumentException("targetNNeighbors must be greater than 2");
    }
    mTargetNNeighbors = targetNNeighbors;
  }

// a: float (optional, default null)
//     More specific parameters controlling the embedding. If null these
//     values are set automatically as determined by <code>minDist</code> and
//     <code>spread</code>.
// b: float (optional, default null)
//     More specific parameters controlling the embedding. If null these
//     values are set automatically as determined by <code>minDist</code> and
//     <code>spread</code>.

  /**
   * Set weighting factor between data topology and target topology. A value of
   * 0.0 weights entirely on data, a value of 1.0 weights entirely on target.
   * The default of 0.5 balances the weighting equally between data and target.
   * @param targetWeight target weighting factor
   */
  public void setTargetWeight(final float targetWeight) {
    mTargetWeight = targetWeight;
  }

  /**
   * Random seed used for the stochastic aspects of the transform operation.
   * This ensures consistency in transform operations. Default: 42.
   * @param transformSeed random number generator seed
   */
  public void setTransformSeed(final int transformSeed) {
    mTransformSeed = transformSeed;
  }

  private void validateParameters() {
    if (mMinDist > mSpread) {
      throw new IllegalArgumentException("minDist must be less than or equal to spread");
    }
//    if (!isinstance(init, str) && !isinstance(init, np.ndarray)) {
//      throw new IllegalArgumentException("init must be a string or ndarray");
//    }
    if (!"spectral".equals(mInit) && !"random".equals(mInit)) {
      throw new IllegalArgumentException("string init values must be 'spectral' || 'random'");
    }
//    if (isinstance(init, np.ndarray) && init.shape[1] != nComponents) {
//      throw new IllegalArgumentException("init ndarray must match nComponents value");
//    }
  }

  /**
   * Fit instances into an embedded space.
   * Optionally use y for supervised dimension reduction.
   * @param instances array of shape <code>(nSamples, nFeatures)</code> or <code>(nSamples, nSamples)</code>
   * If the metric is 'precomputed' instances must be a square distance
   * matrix. Otherwise it contains a sample per row. If the method
   * is 'exact',
   * @param y array of shape <code>(nSamples)</code>
   * A target array for supervised dimension reduction. How this is
   * handled is determined by parameters UMAP was instantiated with.
   * The relevant metric is <code>mTargetMetric</code>.
   */
  private void fit(Matrix instances, float[] y) {

    if (mVerbose) {
      Utils.message("Starting fitting for " + instances.rows() + " instances with " + instances.cols() + " attributes");
    }

    mRawData = instances;

    // Handle all the optional arguments, setting default
    if (mA == null || mB == null) {
      final float[] ab = findAbParams(mSpread, mMinDist);
      mRunA = ab[0];
      mRunB = ab[1];
    } else {
      mRunA = mA;
      mRunB = mB;
    }

//      if (isinstance(this.init, np.ndarray)) {
//        init = check_array(this.init,        /*  dtype = */np.float32,         /* accept_sparse =*/ false);
//      } else {
//        init = this.init;
//      }
    String init = this.mInit;

    mInitialAlpha = mLearningRate;

    validateParameters();

    // Error check nNeighbors based on data size
    if (instances.rows() <= mNNeighbors) {
      if (instances.rows() == 1) {
        mEmbedding = new DefaultMatrix(new float[1][mNComponents]);
        return;
      }

      Utils.message("nNeighbors is larger than the dataset size; truncating to X.length - 1");
      mRunNNeighbors = instances.rows() - 1;
    } else {
      mRunNNeighbors = mNNeighbors;
    }

    if (instances instanceof CsrMatrix) {   // scipy.sparse.isspmatrix_csr(X)) {
      final CsrMatrix csrInstances = (CsrMatrix) instances;
      if (!csrInstances.hasSortedIndices()) {
        csrInstances.sortIndices();
      }
      mSparseData = true;
    } else {
      mSparseData = false;
    }

    if (mVerbose) {
      Utils.message("Construct fuzzy simplicial set: " + instances.rows());
    }

    // Handle small cases efficiently by computing all distances
    if (instances.rows() < SMALL_PROBLEM_THRESHOLD) {
      mSmallData = true;
      final Matrix dmat = PairwiseDistances.pairwiseDistances(instances, mMetric);
      mGraph = fuzzySimplicialSet(dmat, mRunNNeighbors, mRandom, PrecomputedMetric.SINGLETON, null, null, mAngularRpForest, mSetOpMixRatio, mLocalConnectivity, mVerbose);
    } else {
      mSmallData = false;
      // Standard case
      final IndexedDistances nn = nearestNeighbors(instances, mRunNNeighbors, mMetric, mAngularRpForest, mRandom, mVerbose);
      mKnnIndices = nn.getIndices();
      mKnnDists = nn.getDistances();
      mRpForest = nn.getForest();

      mGraph = fuzzySimplicialSet(instances, mNNeighbors, mRandom, mMetric, mKnnIndices, mKnnDists, mAngularRpForest, mSetOpMixRatio, mLocalConnectivity, mVerbose);

      mDistanceFunc = mMetric;
      if (mMetric == PrecomputedMetric.SINGLETON) {
        Utils.message("Using precomputed metric; transform will be unavailable for new data");
      } else {
        mRandomInit = new NearestNeighborRandomInit(mDistanceFunc);
        mTreeInit = new NearestNeighborTreeInit(mDistanceFunc);
        mSearch = new NearestNeighborSearch(mDistanceFunc);
      }
    }

    if (y != null) {
      if (instances.length() != y.length) {
        throw new IllegalArgumentException("Length of x =  " + instances.length() + ", length of y = " + y.length + ", while it must be equal.");
      }
      if (CategoricalMetric.SINGLETON.equals(mTargetMetric)) {
        final float farDist = mTargetWeight < 1.0 ? 2.5F * (1.0F / (1.0F - mTargetWeight)) : 1.0e12F;
        mGraph = categoricalSimplicialSetIntersection((CooMatrix) mGraph, y, farDist);
      } else {
        final int targetNNeighbors = mTargetNNeighbors == -1 ? mRunNNeighbors : mTargetNNeighbors;

        Matrix targetGraph;
        // Handle the small case as precomputed as before
        if (y.length < SMALL_PROBLEM_THRESHOLD) {
          final Matrix ydmat = PairwiseDistances.pairwiseDistances(MathUtils.promoteTranspose(y), mTargetMetric);
          targetGraph = fuzzySimplicialSet(ydmat, targetNNeighbors, mRandom, PrecomputedMetric.SINGLETON, null, null, false, 1.0F, 1, false);
        } else {
          // Standard case
          targetGraph = fuzzySimplicialSet(MathUtils.promoteTranspose(y), targetNNeighbors, mRandom, mTargetMetric, null, null, false, 1.0F, 1, false);
        }
        mGraph = generalSimplicialSetIntersection(mGraph, targetGraph, mTargetWeight);
        mGraph = resetLocalConnectivity(mGraph);
      }
    }

    final int nEpochs = mNEpochs == null ? 0 : mNEpochs;

    if (mVerbose) {
      Utils.message("Construct embedding");
    }

    mEmbedding = simplicialSetEmbedding(mRawData, mGraph, mNComponents, mInitialAlpha, mRunA, mRunB, mRepulsionStrength, mNegativeSampleRate, nEpochs, init, mRandom, mMetric, mVerbose);

    if (mVerbose) {
      Utils.message("Finished embedding");
    }
  }

  /**
   * Fit instances into an embedded space and return that transformed output.
   * @param instances array of shape <code>(nSamples, nFeatures)</code> or <code>(nSamples, nSamples)</code>
   * If the metric is <code>PrecomputedMetric.SINGLETON</code> instances must be a square distance
   * matrix. Otherwise it contains a sample per row.
   * @param y array of shape <code>(nSamples)</code>
   * A target array for supervised dimension reduction. How this is
   * handled is determined by parameters UMAP was instantiated with.
   * The relevant metric is <code>mTargetMetric</code>.
   * @return array of shape <code>(nSamples, nComponents)</code>
   * Embedding of the training data in low-dimensional space.
   */
  public Matrix fitTransform(final Matrix instances, final float[] y) {
    fit(instances, y);
    return mEmbedding;
  }

  public Matrix fitTransform(final Matrix instances) {
    return fitTransform(instances, null);
  }

  public float[][] fitTransform(final float[][] instances) {
    return fitTransform(new DefaultMatrix(instances), null).toArray();
  }

  /**
   * Transform instances into the existing embedded space and return that
   * transformed output.
   * @param instances array, shape <code>(nSamples, nFeatures)</code>
   * New data to be transformed.
   * @return array, shape <code>(nSamples, nComponents)</code>
   * Embedding of the new data in low-dimensional space.
   * @throws IllegalArgumentException If we fit just a single instance then error.
   */
  private Matrix transform(Matrix instances) {
    if (mEmbedding.rows() == 1) {
      throw new IllegalArgumentException("Transform unavailable when model was fit with only a single data sample.");
    }
    if (mSparseData) {
      throw new IllegalArgumentException("Transform not available for sparse input.");
    } else if (mMetric instanceof PrecomputedMetric) {
      throw new IllegalArgumentException("Transform of new data not available for precomputed metric.");
    }

    int[][] indices;
    float[][] dists;
    if (mSmallData) {
      final Matrix dmat = PairwiseDistances.pairwiseDistances(instances, mRawData, mMetric);
      indices = MathUtils.subarray(MathUtils.argpartition(dmat, mRunNNeighbors), mRunNNeighbors);
      final float[][] dmatShortened = Utils.submatrix(dmat, indices, mRunNNeighbors);
      final int[][] indicesSorted = MathUtils.argsort(dmatShortened);
      indices = Utils.submatrix(indices, indicesSorted, mRunNNeighbors);
      dists = Utils.submatrix(dmatShortened, indicesSorted, mRunNNeighbors);
    } else {
      final Heap init = NearestNeighborDescent.initialiseSearch(mRpForest, mRawData, instances, (int) (mRunNNeighbors * mTransformQueueSize), mRandomInit, mTreeInit, mRandom);
      if (mSearchGraph == null) {
        mSearchGraph = new SearchGraph(mRawData.rows());
        for (int k = 0; k < mKnnIndices.length; ++k) {
          for (int j = 0; j < mKnnIndices[k].length; ++j) {
            if (mKnnDists[k][j] != 0) {
              mSearchGraph.set(k, mKnnIndices[k][j]);
            }
          }
        }
      }
      final Heap result = mSearch.initializedNndSearch(mRawData, mSearchGraph, init, instances).deheapSort();
      indices = MathUtils.subarray(result.indices(), mRunNNeighbors);
      dists = MathUtils.subarray(result.weights(), mRunNNeighbors);
    }

    int adjustedLocalConnectivity = Math.max(0, mLocalConnectivity - 1);
    final float[][] sigmasRhos = smoothKnnDist(dists, mRunNNeighbors, adjustedLocalConnectivity);
    float[] sigmas = sigmasRhos[0];
    float[] rhos = sigmasRhos[1];
    CooMatrix graph = computeMembershipStrengths(indices, dists, sigmas, rhos, instances.rows(), mRawData.rows());

    // This was a very specially constructed graph with constant degree.
    // That lets us do fancy unpacking by reshaping the csr matrix indices
    // and data. Doing so relies on the constant degree assumption!
    //CsrMatrix csr_graph = (CsrMatrix) Normalize.normalize(graph.toCsr(), "l1");
//    int[][] inds = csr_graph.indices.reshape(X.rows(), this._n_neighbors);
//    float[][] weights = csr_graph.data.reshape(X.rows(), this._n_neighbors);
    // todo following need to be "reshape" as above
    int[][] inds = null;
    float[][] weights = null;
    Matrix embedding = initTransform(inds, weights, mEmbedding);

    final int nEpochs;
    if (mNEpochs == null) {
      // For smaller datasets we can use more epochs
      if (graph.rows() <= 10000) {
        nEpochs = 100;
      } else {
        nEpochs = 30;
      }
    } else {
      nEpochs = mNEpochs; // 3.0
    }

    MathUtils.zeroEntriesBelowLimit(graph.data(), MathUtils.max(graph.data()) / (float) nEpochs);
    graph = (CooMatrix) graph.eliminateZeros();

    final float[] epochsPerSample = makeEpochsPerSample(graph.data(), nEpochs);

    int[] head = graph.row();
    int[] tail = graph.col();

    return optimizeLayout(embedding, mEmbedding.copy(), head, tail, nEpochs, graph.cols(), epochsPerSample, mRunA, mRunB, mRandom, mRepulsionStrength, mInitialAlpha, mNegativeSampleRate, mVerbose);
  }
}