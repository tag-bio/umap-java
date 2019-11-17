package com.tagbio.umap.metric;

/**
 * Reduced Euclidean distance.
 * @author Sean A. Irvine
 */
public class ReducedEuclideanMetric extends Metric {

  public static final ReducedEuclideanMetric SINGLETON = new ReducedEuclideanMetric();

  public ReducedEuclideanMetric() {
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
    return result;
  }
}
