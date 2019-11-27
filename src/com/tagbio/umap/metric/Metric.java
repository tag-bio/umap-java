/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap.metric;

import java.util.HashMap;
import java.util.Map;

/**
 * Definition of metrics. Individual subclasses implement specific metrics.
 * A convenience function to select metrics by a string name is also provided.
 * @author Leland McInnes (Python)
 * @author Sean A. Irvine
 * @author Richard Littin
 */
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

  private static final Map<String, Metric> METRICS = new HashMap<>();
  static {
    METRICS.put("euclidean", EuclideanMetric.SINGLETON);
    METRICS.put("l2", EuclideanMetric.SINGLETON);
    METRICS.put("manhattan", ManhattanMetric.SINGLETON);
    METRICS.put("l1", ManhattanMetric.SINGLETON);
    METRICS.put("taxicab", ManhattanMetric.SINGLETON);
    METRICS.put("chebyshev", ChebyshevMetric.SINGLETON);
    METRICS.put("linfinity", ChebyshevMetric.SINGLETON);
    METRICS.put("linfty", ChebyshevMetric.SINGLETON);
    METRICS.put("linf", ChebyshevMetric.SINGLETON);
    METRICS.put("canberra", CanberraMetric.SINGLETON);
    METRICS.put("cosine", CosineMetric.SINGLETON);
    METRICS.put("correlation", CorrelationMetric.SINGLETON);
    METRICS.put("haversine", HaversineMetric.SINGLETON);
    METRICS.put("braycurtis", BrayCurtisMetric.SINGLETON);
    METRICS.put("hamming", HammingMetric.SINGLETON);
    METRICS.put("jaccard", JaccardMetric.SINGLETON);
    METRICS.put("dice", DiceMetric.SINGLETON);
    METRICS.put("matching", MatchingMetric.SINGLETON);
    METRICS.put("kulsinski", KulsinskiMetric.SINGLETON);
    METRICS.put("rogerstanimoto", RogersTanimotoMetric.SINGLETON);
    METRICS.put("russellrao", RussellRaoMetric.SINGLETON);
    METRICS.put("sokalsneath", SokalSneathMetric.SINGLETON);
    METRICS.put("sokalmichener", SokalMichenerMetric.SINGLETON);
    METRICS.put("yule", YuleMetric.SINGLETON);
  }

  /**
   * Retrieve a metric by name.
   * @param name name of metric
   * @return metric
   */
  public static Metric getMetric(final String name) {
    final Metric m = METRICS.get(name.toLowerCase());
    if (m == null) {
      throw new IllegalArgumentException("Unknown metric: " + name);
    }
    return m;
  }
}
