package com.tagbio.umap.metric;

/**
 * Hamming distance.
 */
public class HammingMetric extends Metric {

  public static final HammingMetric SINGLETON = new HammingMetric();

  private HammingMetric() {
    super(false);
  }

  @Override
  public double distance(final float[] x, final float[] y) {
    double result = 0.0;
    for (int i = 0; i < x.length; ++i) {
      if (x[i] != y[i]) {
        ++result;
      }
    }
    return result / x.length;
  }
}
