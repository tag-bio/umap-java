package com.tagbio.umap;

/**
 * Math utilities equivalent to Python numpy functionality.
 * @author Sean A. Irvine
 */
class MathUtils {

  private MathUtils() { }

  private static final double INV_LOG2 = 1.0 / Math.log(2);

  /**
   * Return an array filled with zeros.
   * @param n length of array
   * @return array of zeros
   */
  static float[] zeros(final int n) {
    return new float[n];
  }

  static double log2(final double x) {
    return Math.log(x) * INV_LOG2;
  }

  static float max(final float... x) {
    float max = Float.NEGATIVE_INFINITY;
    for (final float v : x) {
      if (v > max) {
        max = v;
      }
    }
    return max;
  }

  static float min(final float... x) {
    float min = Float.POSITIVE_INFINITY;
    for (final float v : x) {
      if (v < min) {
        min = v;
      }
    }
    return min;
  }

  static double mean(final float[] x) {
    double s = 0;
    for (final float v : x) {
      s += v;
    }
    return s / x.length;
  }

  static double mean(final float[][] x) {
    double s = 0;
    long c = 0;
    for (final float[] row : x) {
      for (final float v : row) {
        s += v;
        ++c;
      }
    }
    return s / c;
  }

  /**
   * Retain only positive members of x in a new array.
   * @param x array
   * @return positives
   */
  static float[] filterPositive(final float[] x) {
    int len = 0;
    for (final float v : x) {
      if (v > 0) {
        ++len;
      }
    }
    final float[] res = new float[len];
    int k = 0;
    for (final float v : x) {
      if (v > 0) {
        res[k++] = v;
      }
    }
    return res;
  }

  static boolean containsNegative(final float[][] x) {
    for (final float[] row : x) {
      for (final float v : row) {
        if (v < 0) {
          return true;
        }
      }
    }
    return false;
  }

  static float[] multiply(final float[] x, final float s) {
    final float[] res = new float[x.length];
    for (int k = 0; k < x.length; ++k) {
      res[k] = x[k] * s;
    }
    return res;
  }

  static float[] divide(final float[] x, final float s) {
    return multiply(x, 1.0F / s);
  }

  // todo test cf. numpy
  static float[] linspace(final float start, final float end, final int n) {
    final float[] res = new float[n];
    final float span = end - start;
    final float step = span / n;
    for (int k = 0; k < res.length; ++k) {
      res[k] = start + k * step;
    }
    return res;
  }

  static int[] argsort(final float[] x) {
    // todo return an array of indices that would sort x (i.e. effectively satellite sort on identity array)
    // todo perhaps do this in another clasee
    // todo note funcitonality exists in some libraries
    return null;
  }

  static void zeroEntriesBelowLimit(final float[] x, final float limit) {
    for (int k = 0; k < x.length; ++k) {
      if (x[k] < limit) {
        x[k] = 0;
      }
    }
  }
}
