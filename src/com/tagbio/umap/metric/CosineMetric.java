package com.tagbio.umap.metric;

/**
 * Cosine distance.
 * @author Sean A. Irvine
 */
public class CosineMetric extends Metric {

  public CosineMetric() {
    super(true);
  }

  @Override
  public double distance(final float[] x, final float[] y) {
    double result = 0.0;
    double norm_x = 0.0;
    double norm_y = 0.0;
    for (int i = 0; i < x.length; ++i) {
      result += x[i] * y[i];
      norm_x += x[i] * x[i];
      norm_y += y[i] * y[i];
    }
    if (norm_x == 0.0 && norm_y == 0.0) {
      return 0.0;
    } else if (norm_x == 0.0 || norm_y == 0.0) {
      return 1.0;
    } else {
      return 1.0 - (result / Math.sqrt(norm_x * norm_y));
    }
  }
}
