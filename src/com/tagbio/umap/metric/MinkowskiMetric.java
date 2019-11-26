package com.tagbio.umap.metric;

/**
 * Minkowski distance.
 */
public class MinkowskiMetric extends Metric {

  private final double mPower;

  public MinkowskiMetric(final double power) {
    super(false);
    mPower = power;
  }

  @Override
  public double distance(final float[] x, final float[] y) {
    // D(x, y) = \left(\sum_i |x_i - y_i|^p\right)^{\frac{1}{p}}
    double result = 0.0;
    for (int i = 0; i < x.length; ++i) {
      result += Math.pow(Math.abs(x[i] - y[i]), mPower);
    }
    return Math.pow(result, 1 / mPower);
  }
}
