package com.tagbio.umap.metric;

/**
 * Matching distance.
 */
public class MatchingMetric extends Metric {

  public static final MatchingMetric SINGLETON = new MatchingMetric();

  private MatchingMetric() {
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
    return numNotEqual / (float) x.length;
  }
}
