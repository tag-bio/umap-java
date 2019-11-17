package com.tagbio.umap.metric;

/**
 * Euclidean distance.
 * @author Sean A. Irvine
 */
public class EuclideanMetric extends Metric {

  public static final EuclideanMetric SINGLETON = new EuclideanMetric();

  public EuclideanMetric() {
    super(false);
  }

  @Override
  public double distance(final float[] x, final float[] y) {
    //  D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
    double result = 0.0;
    for (int i = 0; i < x.length; ++i) {
      final double d = x[i] - y[i];
      result += d * d;
    }
    return Math.sqrt(result);
  }
}
