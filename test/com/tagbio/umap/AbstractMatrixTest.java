/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap;

import java.util.Arrays;

import junit.framework.TestCase;

/**
 * Tests the corresponding class.
 * @author Sean A. Irvine
 */
public abstract class AbstractMatrixTest extends TestCase {

  // [[0,1], [1/2,2], [1,0], [0,3]]
  abstract Matrix getMatrixA();

  public void testShape() {
    final Matrix m = getMatrixA();
    assertEquals(2, m.shape().length);
    assertEquals(4, m.shape()[0]);
    assertEquals(2, m.shape()[1]);
    assertEquals(8, m.length());
  }

  public void testToString() {
    assertEquals("0.0,1.0\n0.5,2.0\n1.0,0.0\n0.0,3.0\n", getMatrixA().toString());
  }

  public void testEquals() {
    final Matrix m = getMatrixA();
    assertEquals(m, m);
    assertEquals(m, m.toCoo());
    assertEquals(m, m.toCsr());
    assertFalse(m.equals(null));
    assertFalse(m.equals(m.transpose()));
  }

  public void testRow() {
    assertTrue(Arrays.equals(new float[] {0, 1}, getMatrixA().row(0)));
    assertTrue(Arrays.equals(new float[] {0.5F, 2}, getMatrixA().row(1)));
  }

  public void testTranspose() {
    assertEquals("0.0,0.5,1.0,0.0\n1.0,2.0,0.0,3.0\n", getMatrixA().transpose().toString());
  }

  public void testAdd() {
    assertEquals("0.0,2.0\n1.0,4.0\n2.0,0.0\n0.0,6.0\n", getMatrixA().add(getMatrixA()).toString());
    try {
      getMatrixA().add(getMatrixA().transpose());
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  public void testSubtract() {
    assertEquals("0.0,0.0\n0.0,0.0\n0.0,0.0\n0.0,0.0\n", getMatrixA().subtract(getMatrixA()).toString());
    try {
      getMatrixA().subtract(getMatrixA().transpose());
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  public void testMultiplyScalar() {
    assertEquals("0.0,3.0\n1.5,6.0\n3.0,0.0\n0.0,9.0\n", getMatrixA().multiply(3).toString());
  }

  public void testMultiply() {
    final Matrix m = getMatrixA();
    try {
      m.multiply(m);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
    final Matrix mt = m.transpose();
    assertEquals("1.0,2.0,0.0,3.0\n2.0,4.25,0.5,6.0\n0.0,0.5,1.0,0.0\n3.0,6.0,0.0,9.0\n", m.multiply(mt).toString());
    assertEquals("1.25,1.0\n1.0,14.0\n", mt.multiply(m).toString());
  }

  public void testGet() {
    final Matrix m = getMatrixA();
    assertEquals(0.0f, m.get(0, 0));
    assertEquals(1.0f, m.get(0, 1));
    assertEquals(0.5f, m.get(1, 0));
    assertEquals(2.0f, m.get(1, 1));
    assertEquals(1.0f, m.get(2, 0));
    assertEquals(0.0f, m.get(2, 1));
    assertEquals(0.0f, m.get(3, 0));
    assertEquals(3.0f, m.get(3, 1));
  }

  public void testHadamardMultiply() {
    final Matrix m = getMatrixA();
    final Matrix res = m.hadamardMultiply(m);
    assertTrue(Arrays.equals(m.shape(), res.shape()));
    for (int r = 0; r < res.rows(); ++r) {
      for (int c = 0; c < res.cols(); ++c) {
        assertEquals(m.get(r, c) * m.get(r, c), res.get(r, c));
      }
    }
  }

  public void testRowNormalize() {
    assertEquals("0.0,1.0\n0.25,1.0\n1.0,0.0\n0.0,1.0\n", getMatrixA().rowNormalize().toString());
  }
}
