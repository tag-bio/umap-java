package com.tagbio.umap.metric;

/**
 * Special indicator for categorical data.
 * @author Sean A. Irvine
 */
public class CategoricalMetric extends Metric {

  public static final CategoricalMetric SINGLETON = new CategoricalMetric();

  private CategoricalMetric() {
    super(false);
  }

  @Override
  public double distance(final float[] x, final float[] y) {
    throw new IllegalStateException();
  }
}
