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
