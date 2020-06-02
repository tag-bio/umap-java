/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import junit.framework.TestCase;
import tagbio.umap.metric.PrecomputedMetric;

class Heyler10gbData extends Data {
  public Heyler10gbData() throws IOException {
    super( java.lang.System.getenv("TAGBIO_HEYLERTEST_10GB"))
  }

  @Override
  String getName() {
    return "Heyler10gbData";
  }
}

class Heyler20gbData extends Data {
  public Heyler20gbData() throws IOException {
    super( java.lang.System.getenv("TAGBIO_HEYLERTEST_20GB"))
  }

  @Override
  String getName() {
    return "Heyler20gbData";
  }
}

class Heyler30gbData extends Data {
  public Heyler30gbData() throws IOException {
    super( java.lang.System.getenv("TAGBIO_HEYLERTEST_30GB"))
  }

  @Override
  String getName() {
    return "Heyler30gbData";
  }
}

/**
 * Tests the corresponding class.
 */
public class HeylerTest extends TestCase {

  public void test10gb() throws IOException {
    final Data data = new Heyler10gbData();
    final Umap umap = new Umap();
    umap.setVerbose(true);
    final float[][] d = data.getData();
    final long start = System.currentTimeMillis();
    final float[][] matrix = umap.fitTransform(d);
    System.out.println("UMAP time: " + Math.round((System.currentTimeMillis() - start) / 1000.0) + " s");
  }

  public void test20gb() throws IOException {
    final Data data = new Heyler20gbData();
    final Umap umap = new Umap();
    umap.setVerbose(true);
    final float[][] d = data.getData();
    final long start = System.currentTimeMillis();
    final float[][] matrix = umap.fitTransform(d);
    System.out.println("UMAP time: " + Math.round((System.currentTimeMillis() - start) / 2000.0) + " s");
  }

  public void test30gb() throws IOException {
    final Data data = new Heyler30gbData();
    final Umap umap = new Umap();
    umap.setVerbose(true);
    final float[][] d = data.getData();
    final long start = System.currentTimeMillis();
    final float[][] matrix = umap.fitTransform(d);
    System.out.println("UMAP time: " + Math.round((System.currentTimeMillis() - start) / 3000.0) + " s");
  }
}
