package com.tagbio.umap;

import junit.framework.TestCase;

public class IrisDataTest extends TestCase {
  public void testData() {
    final IrisData irisData = new IrisData();
    final String[] targetNames = irisData.getTargetNames();
    assertEquals(3, targetNames.length);
    assertEquals("setosa", targetNames[0]);
    final int[] targets = irisData.getTargets();
    assertEquals(150, targets.length);
    for (int target : targets) {
      assertTrue(target >= 0);
      assertTrue(target < 3);
    }
    final float[][] data = irisData.getData();
    assertEquals(150, data.length);
    assertEquals(4, data[0].length);
    for (float[] row : data) {
      for (float val : row) {
        assertTrue("Value < 0: " + val,val >= 0.0F);
        assertTrue("Value > 7.9: " + val,val <= 7.9F);
      }
    }
  }
}
