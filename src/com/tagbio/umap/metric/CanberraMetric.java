package com.tagbio.umap.metric;

/**
 * Canberra distance.
 */
public class CanberraMetric extends Metric {

  public static final CanberraMetric SINGLETON = new CanberraMetric();

  private CanberraMetric() {
    super(false);
  }

  @Override
  public double distance(final float[] x, final float[] y) {
    double result = 0.0;
    for (int i = 0; i < x.length; ++i) {
      final double denominator = Math.abs(x[i]) + Math.abs(y[i]);
      if (denominator > 0) {
        result += Math.abs(x[i] - y[i]) / denominator;
      }
    }
    return result;
  }
}
