package com.tagbio.umap;

import junit.framework.TestCase;

public class MathUtilsTest extends TestCase {
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
