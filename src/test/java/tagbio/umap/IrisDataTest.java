/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.io.IOException;

import junit.framework.TestCase;

public class IrisDataTest extends TestCase {
  public void testData() throws IOException {
    final IrisData irisData = new IrisData();
    final String[] attributes = irisData.getAttributes();
    assertEquals(4, attributes.length);
    for (String att : attributes) {
      assertTrue(att, att.matches("att[0-9]+"));
    }
    final String[] names = irisData.getSampleNames();
    assertEquals(150, names.length);
    for (String name : names) {
      assertTrue(name, name.matches("[a-z]*:[0-9]+"));
    }
    final float[][] data = irisData.getData();
    assertEquals(150, data.length);
    assertEquals(4, data[0].length);
    for (float[] row : data) {
      for (float val : row) {
        assertTrue("Value < 0: " + val, val >= 0.0F);
        assertTrue("Value > 16: " + val, val <= 16.0F);
      }
    }
  }

  public void testDataSmall() throws IOException {
    final IrisData irisData = new IrisData(true);
    final String[] attributes = irisData.getAttributes();
    assertEquals(4, attributes.length);
    for (String att : attributes) {
      assertTrue(att, att.matches("att[0-9]+"));
    }
    final String[] names = irisData.getSampleNames();
    assertEquals(10, names.length);
    for (String name : names) {
      assertTrue(name, name.matches("[a-z]*:[0-9]+"));
    }
    final float[][] data = irisData.getData();
    assertEquals(10, data.length);
    assertEquals(4, data[0].length);
    for (float[] row : data) {
      for (float val : row) {
        assertTrue("Value < 0: " + val, val >= 0.0F);
        assertTrue("Value > 16: " + val, val <= 16.0F);
      }
    }
  }
}
