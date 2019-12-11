package tagbio.umap;

import java.util.Random;
import java.util.TreeSet;

import junit.framework.TestCase;

public class UtilsTest extends TestCase {

  public void testRejectionSample() {
    final Random r = new Random();
    final int[] rs = Utils.rejectionSample(20, 100, r);
    assertEquals(20, rs.length);
    final TreeSet<Integer> uniq = new TreeSet<>();
    for (final int v : rs) {
      assertTrue(v >= 0 && v < 100);
      uniq.add(v);
    }
    assertEquals(20, uniq.size());
    try {
      Utils.rejectionSample(5, 2, r);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }
}

