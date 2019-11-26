package com.tagbio.umap.metric;

/**
 * Weighted Minkowski distance.
 */
public class WeightedMinkowskiMetric extends Metric {

  private final double mPower;
  private final float[] mWeights;

  public WeightedMinkowskiMetric(final double power, final float[] weights) {
    super(false);
    mPower = power;
    mWeights = weights;
  }

  @Override
  public double distance(final float[] x, final float[] y) {
    // D(x, y) = \left(\sum_i w_i |x_i - y_i|^p\right)^{\frac{1}{p}}
    double result = 0.0;
    for (int i = 0; i < x.length; ++i) {
      result += Math.pow(mWeights[i] * Math.abs(x[i] - y[i]), mPower);
    }
    return Math.pow(result, 1 / mPower);
  }
}
