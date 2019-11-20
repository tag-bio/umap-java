package com.tagbio.umap;

import java.util.Random;

import junit.framework.TestCase;

/**
 * Tests the corresponding class.
 *
 * @author Jonathan Purvis
 * @author Sean A. Irvine
 */
public class SortTest extends TestCase {

  /**
   * Checks that an array is sorted.
   * @param a primary array
   * @return <code>true</code> if the array is sorted, <code>false</code> otherwise
   */
  static boolean isSorted(final float[] a) {
    for (int ii = 0; ii < a.length - 1; ++ii) {
      if (a[ii] > a[ii + 1]) {
        return false;
      }
    }
    return true;
  }

  public void testSortAtRandomPairsD() {
    final Random r = new Random();
    for (int i = 0; i < 5; ++i) {
      final int l = r.nextInt(1000);
      final float[] keys = new float[l];
      final int[] pairs = new int[l];
      // init arrays to be the same
      for (int j = 0; j < l; ++j) {
        keys[j] = r.nextFloat();
        pairs[j] = j;
      }
      Sort.sort(keys, pairs);
      assertTrue(isSorted(keys));
      long s = 0;
      for (int j = 0; j < l; ++j) {
        s += pairs[j];
      }
      assertEquals(l * (l - 1) / 2, s);
    }
  }

  public void testMedD() {
    final float[] a = {0, 16, 256, 409};
    assertEquals(2, Sort.med3(a, 1, 2, 3));
    assertEquals(2, Sort.med3(a, 1, 3, 2));
    assertEquals(2, Sort.med3(a, 2, 1, 3));
    assertEquals(2, Sort.med3(a, 2, 3, 1));
    assertEquals(2, Sort.med3(a, 3, 1, 2));
    assertEquals(2, Sort.med3(a, 3, 2, 1));
  }
}
