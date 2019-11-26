package com.tagbio.umap;

import java.io.IOException;

import junit.framework.TestCase;

public class DigitDataTest extends TestCase {
  public void testData() throws IOException {
    final DigitData digitData = new DigitData();
    final String[] attributes = digitData.getAttributes();
    assertEquals(64, attributes.length);
    for (String att : attributes) {
      assertTrue(att, att.matches("att[0-9]+"));
    }
    final String[] names = digitData.getSampleNames();
    assertEquals(1797, names.length);
    for (String name : names) {
      assertTrue(name, name.matches("[0-9]:[0-9]+"));
    }
    final float[][] data = digitData.getData();
    assertEquals(1797, data.length);
    assertEquals(64, data[0].length);
    for (float[] row : data) {
      for (float val : row) {
        assertTrue("Value < 0: " + val, val >= 0.0F);
        assertTrue("Value > 16: " + val, val <= 16.0F);
      }
    }
  }
}
