package com.tagbio.umap.metric;

/**
 * Euclidean distance standardised against a vector of standard deviations per coordinate.
 * @author Sean A. Irvine
 */
public class StandardisedEuclideanMetric extends Metric {

  private final float[] mSigma;

  public StandardisedEuclideanMetric(final float[] sigma) {
    super(false);
    mSigma = sigma;
  }

  @Override
  public double distance(final float[] x, final float[] y) {
    //  D(x, y) = \sqrt{\sum_i \frac{(x_i - y_i)**2}{v_i}}
    double result = 0.0;
    for (int i = 0; i < x.length; ++i) {
      final double d = x[i] - y[i];
      result += d * d / mSigma[i];
    }
    return Math.sqrt(result);
  }
}
