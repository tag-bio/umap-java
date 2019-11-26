package com.tagbio.umap.metric;

/**
 * Manhattan distance.
 */
public class ManhattanMetric extends Metric {

  public static final ManhattanMetric SINGLETON = new ManhattanMetric();

  private ManhattanMetric() {
    super(false);
  }

  @Override
  public double distance(final float[] x, final float[] y) {
    //  D(x, y) = \sum_i |x_i - y_i|
    double result = 0.0;
    for (int i = 0; i < x.length; ++i) {
      result += Math.abs(x[i] - y[i]);
    }
    return result;
  }
}
