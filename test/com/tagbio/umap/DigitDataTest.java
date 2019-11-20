package com.tagbio.umap;

import junit.framework.TestCase;

public class DigitDataTest extends TestCase {
    public void testData() {
    final DigitData digitData = new DigitData();
    final String[] targetNames = digitData.getTargetNames();
    assertEquals(10, targetNames.length);
    assertEquals("0", targetNames[0]);
    final int[] targets = digitData.getTargets();
    assertEquals(1797, targets.length);
    for (int target : targets) {
      assertTrue(target >= 0);
      assertTrue(target <= 9);
    }
    final float[][] data = digitData.getData();
    assertEquals(1797, data.length);
    assertEquals(64, data[0].length);
    for (float[] row : data) {
      for (float val : row) {
        assertTrue("Value < 0: " + val,val >= 0.0F);
        assertTrue("Value > 16: " + val, val <= 16.0F);
      }
    }
  }
}
