/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap.metric;

/**
 * Euclidean distance.
 */
public class EuclideanMetric extends Metric {

  public static final EuclideanMetric SINGLETON = new EuclideanMetric();

  private EuclideanMetric() {
    super(false);
  }

  @Override
  public double distance(final float[] x, final float[] y) {
    //  D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
    float result = 0;
    for (int i = 0; i < x.length; ++i) {
      final float d = x[i] - y[i];
      result += d * d;
    }
    return Math.sqrt(result);
  }
}
