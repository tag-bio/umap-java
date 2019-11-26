package com.tagbio.umap.metric;

/**
 * Bray Curtis distance.
 */
public class BrayCurtisMetric extends Metric {

  public static final BrayCurtisMetric SINGLETON = new BrayCurtisMetric();

  private BrayCurtisMetric() {
    super(false);
  }

  @Override
  public double distance(final float[] x, final float[] y) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (int i = 0; i < x.length; ++i) {
      numerator += Math.abs(x[i] - y[i]);
      denominator += Math.abs(x[i] + y[i]);
    }
    return denominator > 0.0 ? numerator / denominator : 0.0;
  }
}
