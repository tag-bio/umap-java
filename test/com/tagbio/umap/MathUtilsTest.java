package com.tagbio.umap;

import java.util.Arrays;

import junit.framework.TestCase;

public class MathUtilsTest extends TestCase {

  public void testLog2() {
    assertEquals(0, MathUtils.log2(1), 1e-10);
    assertEquals(1, MathUtils.log2(2), 1e-10);
    assertEquals(4, MathUtils.log2(16), 1e-10);
    assertEquals(1.6514961294723187, MathUtils.log2(Math.PI), 1e-10);
  }

  public void testMax() {
    assertEquals(42, MathUtils.max(42), 1e-10);
    assertEquals(42, MathUtils.max(42, 1), 1e-10);
    assertEquals(42, MathUtils.max(1, 42), 1e-10);
    assertEquals(42, MathUtils.max(0, 42, -300.5F), 1e-10);
    assertEquals(Float.NEGATIVE_INFINITY, MathUtils.max());
  }

  public void testMin() {
    assertEquals(42, MathUtils.min(42), 1e-10);
    assertEquals(1, MathUtils.min(42, 1), 1e-10);
    assertEquals(1, MathUtils.min(1, 42), 1e-10);
    assertEquals(-300.5F, MathUtils.min(0, 42, -300.5F), 1e-10);
    assertEquals(Float.POSITIVE_INFINITY, MathUtils.min());
  }

  public void testMean() {
    assertEquals(42, MathUtils.mean(42), 1e-10);
    assertEquals(21, MathUtils.mean(42, 0), 1e-10);
    assertEquals(21, MathUtils.mean(0, 42), 1e-10);
    assertEquals(2.5, MathUtils.mean(0, 1, 2, 3, 4, 5), 1e-10);
  }

  public void testMean2D() {
    assertEquals(7.833333333333333, MathUtils.mean(new float[][] {{0, 42}, {2, 3}, {-7, 7}}), 1e-10);
  }

  public void testFilterPositive() {
    assertTrue(Arrays.equals(new float[0], MathUtils.filterPositive()));
    assertTrue(Arrays.equals(new float[0], MathUtils.filterPositive(-0.1F, -1F, Float.NEGATIVE_INFINITY)));
    assertTrue(Arrays.equals(new float[] {1, 42}, MathUtils.filterPositive(-7, 1, 0, -42, 42)));
  }

  public void testLinspace() {
    float[] res = MathUtils.linspace(2, 5, 15);
    assertNotNull(res);
    assertEquals(15, res.length);
    assertEquals(2.0F, res[0]);
    assertEquals(5.0F, res[res.length - 1]);
    for (int i = 0; i < res.length; i++) {
      assertEquals(2.0F + 3.0F * i / 14.0F, res[i]);
    }
  }
}
