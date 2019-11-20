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
    suite.addTestSuite(SortTest.class);
    suite.addTestSuite(UmapTest.class);
    return suite;
  }

  public static void main(final String[] args) {
    junit.textui.TestRunner.run(suite());
  }
}
