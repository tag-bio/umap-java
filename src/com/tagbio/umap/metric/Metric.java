package com.tagbio.umap.metric;

// # Author: Leland McInnes <leland.mcinnes@gmail.com>
// #
// # License: BSD 3 clause

public abstract class Metric {

  private final boolean mIsAngular;

  Metric(final boolean isAngular) {
    mIsAngular = isAngular;
  }


  /**
   * Distance metric.
   * @param x first point
   * @param y second point
   * @return distance between the points
   */
  public abstract double distance(final float[] x, final float[] y);

  /**
   * Is this an angular metric.
   * @return true iff this metric is angular.
   */
  public boolean isAngular() {
    return mIsAngular;
  }

  // todo move rest of these to subclasses a la EuclideanMetric



// @numba.njit()
// def sokal_michener(x, y):
//     num_not_equal = 0.0
//     for i in range(x.shape[0]):
//         x_true = x[i] != 0
//         y_true = y[i] != 0
//         num_not_equal += x_true != y_true

//     return (2.0 * num_not_equal) / (x.shape[0] + num_not_equal)


// @numba.njit()
// def sokal_sneath(x, y):
//     num_true_true = 0.0
//     num_not_equal = 0.0
//     for i in range(x.shape[0]):
//         x_true = x[i] != 0
//         y_true = y[i] != 0
//         num_true_true += x_true and y_true
//         num_not_equal += x_true != y_true

//     if num_not_equal == 0.0:
//         return 0.0
//     else:
//         return num_not_equal / (0.5 * num_true_true + num_not_equal)


// @numba.njit()
// def haversine(x, y):
//     if x.shape[0] != 2:
//         throw new IllegalArgumentException("haversine is only defined for 2 dimensional data")
//     sin_lat = np.sin(0.5 * (x[0] - y[0]))
//     sin_long = np.sin(0.5 * (x[1] - y[1]))
//     result = np.sqrt(sin_lat ** 2 + np.cos(x[0]) * np.cos(y[0]) * sin_long ** 2)
//     return 2.0 * np.arcsin(result)


// @numba.njit()
// def yule(x, y):
//     num_true_true = 0.0
//     num_true_false = 0.0
//     num_false_true = 0.0
//     for i in range(x.shape[0]):
//         x_true = x[i] != 0
//         y_true = y[i] != 0
//         num_true_true += x_true and y_true
//         num_true_false += x_true and (not y_true)
//         num_false_true += (not x_true) and y_true

//     num_false_false = x.shape[0] - num_true_true - num_true_false - num_false_true

//     if num_true_false == 0.0 || num_false_true == 0.0:
//         return 0.0
//     else:
//         return (2.0 * num_true_false * num_false_true) / (
//             num_true_true * num_false_false + num_true_false * num_false_true
//         )


// named_distances = {
//     # general minkowski distances
//     "euclidean": euclidean,
//     "l2": euclidean,
//     "manhattan": manhattan,
//     "taxicab": manhattan,
//     "l1": manhattan,
//     "chebyshev": chebyshev,
//     "linfinity": chebyshev,
//     "linfty": chebyshev,
//     "linf": chebyshev,
//     "minkowski": minkowski,
//     # Standardised/weighted distances
//     "seuclidean": standardised_euclidean,
//     "standardised_euclidean": standardised_euclidean,
//     "wminkowski": weighted_minkowski,
//     "weighted_minkowski": weighted_minkowski,
//     "mahalanobis": mahalanobis,
//     # Other distances
//     "canberra": canberra,
//     "cosine": cosine,
//     "correlation": correlation,
//     "haversine": haversine,
//     "braycurtis": bray_curtis,
//     # Binary distances
//     "hamming": hamming,
//     "jaccard": jaccard,
//     "dice": dice,
//     "matching": matching,
//     "kulsinski": kulsinski,
//     "rogerstanimoto": rogers_tanimoto,
//     "russellrao": russellrao,
//     "sokalsneath": sokal_sneath,
//     "sokalmichener": sokal_michener,
//     "yule": yule,

}
