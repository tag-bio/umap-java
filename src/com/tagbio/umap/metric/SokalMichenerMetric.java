package com.tagbio.umap.metric;

/**
 * Sokal Michener distance.
 */
public class SokalMichenerMetric extends Metric {

  public static final SokalMichenerMetric SINGLETON = new SokalMichenerMetric();

  private SokalMichenerMetric() {
    super(false);
  }

  @Override
  public double distance(final float[] x, final float[] y) {
    int numNotEqual = 0;
    for (int i = 0; i < x.length; ++i) {
      final boolean xTrue = x[i] != 0;
      final boolean yTrue = y[i] != 0;
      if (xTrue != yTrue) {
        ++numNotEqual;
      }
    }
    return (2.0 * numNotEqual) / (float) (x.length + numNotEqual);
  }
}
