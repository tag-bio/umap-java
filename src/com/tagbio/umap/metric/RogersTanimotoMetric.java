package com.tagbio.umap.metric;

/**
 * Rogers Tanimoto distance.
 */
public class RogersTanimotoMetric extends Metric {

  public static final RogersTanimotoMetric SINGLETON = new RogersTanimotoMetric();

  RogersTanimotoMetric() {
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
