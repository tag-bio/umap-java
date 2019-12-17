/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.util.Arrays;

import junit.framework.TestCase;

public class SparseVectorTest extends TestCase {

  public void testCons() {
    try {
      new SparseVector(new int[1], new float[0]);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  public void testScalarDivide() {
    final SparseVector a = new SparseVector(new int[] {0, 1, 2}, new float[] {5, 4, -7});
    a.divide(2);
    assertEquals("[0, 1, 2]", Arrays.toString(a.getIndices()));
    assertEquals("[2.5, 2.0, -3.5]", Arrays.toString(a.getData()));
  }

  public void testNorm() {
    final SparseVector a = new SparseVector(new int[] {0, 1, 2}, new float[] {5, 4, -7});
    assertEquals(9.4868326, a.norm(), 1e-6);
  }

  public void testAdd() {
    final SparseVector a = new SparseVector(new int[] {0, 1, 2}, new float[] {5, 4, -7});
    assertEquals("[0, 1, 2]", Arrays.toString(a.getIndices()));
    assertEquals("[5.0, 4.0, -7.0]", Arrays.toString(a.getData()));
    final SparseVector b = new SparseVector(new int[] {0, 2, 3}, new float[] {3, 7, 2});
    final SparseVector c = a.add(b);
    assertEquals("[0, 1, 3]", Arrays.toString(c.getIndices()));
    assertEquals("[8.0, 4.0, 2.0]", Arrays.toString(c.getData()));
    assertEquals(14.0, c.sum(), 1e-6);
  }

  public void testSubtract() {
    final SparseVector a = new SparseVector(new int[] {0, 1, 2}, new float[] {5, 4, -7});
    final SparseVector b = new SparseVector(new int[] {0, 2, 3}, new float[] {3, 7, 2});
    final SparseVector c = a.subtract(b);
    assertEquals("[0, 1, 2, 3]", Arrays.toString(c.getIndices()));
    assertEquals("[2.0, 4.0, -14.0, -2.0]", Arrays.toString(c.getData()));
    assertEquals(0.0, a.subtract(a).sum(), 1e-6);
    assertEquals(0, a.subtract(a).getIndices().length);
  }

  public void testMultiply() {
    final SparseVector a = new SparseVector(new int[] {0, 1, 2}, new float[] {5, 4, -7});
    final SparseVector b = new SparseVector(new int[] {0, 2, 3}, new float[] {3, 7, 2});
    final SparseVector c = a.hadamardMultiply(b);
    assertEquals("[0, 2]", Arrays.toString(c.getIndices()));
    assertEquals("[15.0, -49.0]", Arrays.toString(c.getData()));
    assertEquals(0.0, a.subtract(a).sum(), 1e-6);
    assertEquals(0, a.subtract(a).getIndices().length);
  }
}

