package com.tagbio.umap;

/**
 * Tests the corresponding class.
 * @author Sean A. Irvine
 */
public class DefaultMatrixTest extends AbstractMatrixTest {

  Matrix getMatrixA() {
    return new DefaultMatrix(new float[][] {{0, 1}, {0.5F, 2}, {1, 0}, {0, 3}});
  }

  public void testSet() {
    final DefaultMatrix m = new DefaultMatrix(1, 1);
    m.set(0, 0, 42);
    assertEquals(42.0, m.get(0, 0), 1e-10);
  }
}
