package com.tagbio.umap.metric;

/**
 * Dice distance.
 * @author Sean A. Irvine
 */
public class DiceMetric extends Metric {

  public static final DiceMetric SINGLETON = new DiceMetric();

  private DiceMetric() {
    super(true);
  }

  @Override
  public double distance(final float[] x, final float[] y) {
    int numTrueTrue = 0;
    int numNotEqual = 0;
    for (int i = 0; i < x.length; ++i) {
      final boolean xTrue = x[i] != 0;
      final boolean yTrue = y[i] != 0;
      numTrueTrue += xTrue && yTrue ? 1 : 0;
      numNotEqual += xTrue != yTrue ? 1 : 0;
    }

    if (numNotEqual == 0) {
      return 0.0;
    } else {
      return numNotEqual / (2.0 * numTrueTrue + numNotEqual);
    }
  }
}
