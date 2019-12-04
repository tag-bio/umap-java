package com.tagbio.umap;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;

public class TimingTest {

  private static void save(Matrix embedding, int[] indexes, String title) throws FileNotFoundException {
    final File csvOutputFile = new File("/home/richard/tmp/umap/"+ title +".tsv");
    try (PrintWriter pw = new PrintWriter(csvOutputFile)) {
      for (int r = 0; r < embedding.rows(); r++) {
        for (int c = 0; c < embedding.cols(); c++) {
          pw.print(embedding.get(r, c));
          pw.print("\t");
        }
        pw.println(indexes[r]);
      }
    }
  }

  public static void main(final String[] args) {
    try {
      for (Data data : new Data[]{new IrisData(), new DigitData(), new MammothData(), new GeneData()}) {
//      for (Data data : new Data[]{new GeneData()}) {
        System.out.println("DATA: " + data.getName());
        for (int seed : new int[]{42, 123, 98765, -4444, 10101}) {
          for (float minDist : new float[]{0.1f, 0.5f, 0.9f}) {
            for (int neighbours : new int[]{15, 5, 10, 20, 50, 100}) {
              System.out.print(data.getName() + "\t" + minDist + "\t" + neighbours + "\t" + seed);

              final Umap umap = new Umap();
              umap.setInit("random");
              umap.setSeed(seed);
              umap.setMinDist(minDist);
              umap.setNumberNearestNeighbours(neighbours);
              long start = System.nanoTime();
              final Matrix embedding = umap.fitTransform(data.getData());
              long end = System.nanoTime();
              final String title = String.format("%s_md%1.2f_nn%03d_s%d", data.getName(), minDist, neighbours, seed);
              save(embedding, data.getSampleClassIndex(), title);
              System.out.print(String.format("\t%.3f", (end - start) / 1000000000.0F));
            }
          }
        }

//        for (float minDist : new float[]{0.0f, 0.01f, 0.1f, 0.2f, 0.5f, 0.9f, 0.99f, 1.0f}) {
//          System.out.print(minDist);
//          for (int seed : new int[]{/*123, 98765,*/ -4444, 10101, 42}) {
//            final Umap umap = new Umap();
//            umap.setInit("random");
//            umap.setMinDist(minDist);
//            long start = System.nanoTime();
//            final Matrix embedding = umap.fitTransform(data.getData());
//            long end = System.nanoTime();
//            System.out.print(String.format("\t%.3f", (end - start) / 1000000000.0));
//          }
//          System.out.println();
//        }
      }
    } catch (IOException e) {
      e.printStackTrace();
    }

  }

}
