package com.tagbio.umap;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.zip.GZIPInputStream;

abstract class Data {
  private float[][] mData;
  private final List<String> mAttributes = new ArrayList<>();
  private final List<String> mSampleNames = new ArrayList<>();

  Data(String dataFile) throws IOException {
    InputStream is = new BufferedInputStream(getClass().getClassLoader().getResourceAsStream(dataFile));
    if (dataFile.endsWith(".gz")) {
      is = new GZIPInputStream(is);
    }

    final List<float[]> records = new ArrayList<>();

    try (final Scanner scanner = new Scanner(is)) {
      if (scanner.hasNextLine()) {
        // header line
        final String line = scanner.nextLine().trim();
        try (final Scanner rowScanner = new Scanner(line)) {
          rowScanner.useDelimiter("\t");
          if (rowScanner.hasNext()) {
            final String next = rowScanner.next();
            assert ("sample".equals(next));
          }
          while (rowScanner.hasNext()) {
            mAttributes.add(rowScanner.next());
          }
        }
      }
      while (scanner.hasNextLine()) {
        final String line = scanner.nextLine().trim();
        final float[] values = new float[mAttributes.size()];
        try (final Scanner rowScanner = new Scanner(line)) {
          rowScanner.useDelimiter("\t");
          if (rowScanner.hasNext()) {
            mSampleNames.add(rowScanner.next().trim());
          }
          int k = 0;
          while (rowScanner.hasNext()) {
            values[k++] = Float.parseFloat(rowScanner.next());
          }
        }
        records.add(values);
        if (records.size() % 100 == 0) {
          System.out.println("Done reading: " + records.size());
        }
      }
    }
    mData = records.toArray(new float[0][]);
  }

  public float[][] getData() {
    return mData;
  }

  public String[] getAttributes() {
    return mAttributes.toArray(new String[0]);
  }

  public String[] getSampleNames() {
    return mSampleNames.toArray(new String[0]);
  }
}

