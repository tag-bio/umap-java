/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap.metric;

/**
 * Cosine distance.
 * @author Sean A. Irvine
 */
public class CosineMetric extends Metric {

  public static final CosineMetric SINGLETON = new CosineMetric();

  private CosineMetric() {
    super(true);
  }

  @Override
  public float distance(final float[] x, final float[] y) {
    double result = 0.0;
    double norm_x = 0.0;
    double norm_y = 0.0;
    for (int i = 0; i < x.length; ++i) {
      result += x[i] * y[i];
      norm_x += x[i] * x[i];
      norm_y += y[i] * y[i];
    }
    if (norm_x == 0.0 && norm_y == 0.0) {
      return 0;
    } else if (norm_x == 0.0 || norm_y == 0.0) {
      return 1;
    } else {
      return (float) (1 - (result / Math.sqrt(norm_x * norm_y)));
    }
  }
}
