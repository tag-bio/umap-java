/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap;

import junit.framework.Test;
import junit.framework.TestSuite;

/**
 * Links all the tests in this package.
 * @author Sean A. Irvine
 */
public class AllTests extends TestSuite {

  public static Test suite() {
    final TestSuite suite = new TestSuite();
    suite.addTestSuite(CooMatrixTest.class);
    suite.addTestSuite(CsrMatrixTest.class);
    suite.addTestSuite(DefaultMatrixTest.class);
    suite.addTestSuite(DigitDataTest.class);
    suite.addTestSuite(IrisDataTest.class);
    suite.addTestSuite(SortTest.class);
    suite.addTestSuite(UmapTest.class);
    suite.addTestSuite(UtilsTest.class);
    return suite;
  }

  public static void main(final String[] args) {
    junit.textui.TestRunner.run(suite());
  }
}
