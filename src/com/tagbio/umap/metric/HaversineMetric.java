package com.tagbio.umap.metric;

/**
 * Haversine distance.
 */
public class HaversineMetric extends Metric {

  public static final HaversineMetric SINGLETON = new HaversineMetric();

  private HaversineMetric() {
    super(false);
  }

  @Override
  public double distance(final float[] x, final float[] y) {
    if (x.length != 2) {
      throw new IllegalArgumentException("haversine is only defined for 2 dimensional data");
    }
    final double sinLat = Math.sin(0.5 * (x[0] - y[0]));
    final double sinLong = Math.sin(0.5 * (x[1] - y[1]));
    final double result = Math.sqrt(sinLat * sinLat + Math.cos(x[0]) * Math.cos(y[0]) * sinLong * sinLong);
    return 2.0 * Math.asin(result);
  }
}
