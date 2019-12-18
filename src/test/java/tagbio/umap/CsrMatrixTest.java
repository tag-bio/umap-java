/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

/**
 * Tests the corresponding class.
 * @author Sean A. Irvine
 */
public class CsrMatrixTest extends AbstractMatrixTest {

  Matrix getMatrixA() {
    return new DefaultMatrix(new float[][] {{0, 1}, {0.5F, 2}, {1, 0}, {0, 3}}).toCsr();
  }

  // Don't test functionality not yet supported in Csr
  @Override
  public void testAdd() {
  }

  @Override
  public void testSubtract() {
  }

  @Override
  public void testEquals() {
  }

  @Override
  public void testMultiply() {
  }

  @Override
  public void testTranspose() {
  }
}
