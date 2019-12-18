/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class VivianaData extends Data {

  private int[] mClassIndexes;

  public VivianaData() throws IOException {
    super("/everest/Tag-bio/viviana_t_data.csv.gz");
    setSampleNamesFromInfo("tissue");
  }

  @Override
  String getName() {
    return "viviana";
  }

  @Override
  public int[] getSampleClassIndex() {
    return Arrays.copyOf(mClassIndexes, mClassIndexes.length);
  }

  private static String trimQuotes(final String text) {
    return text.replaceAll("^\"|\"$", "");
  }

  public void setSampleNamesFromInfo(String columnName) throws IOException {
    final Map<String, Integer> targetIndex = new HashMap<>();
    final Map<String, Integer> nameIndex = new HashMap<>();
    try (final LineNumberReader r = new LineNumberReader(new InputStreamReader(getStream("/everest/Tag-bio/viviana_t_metadata.csv.gz")))) {
      String line = r.readLine();
      if (line == null) {
        throw new IOException("No header line");
      }
      final String[] sampleNames = line.trim().split(",");
      // Process data lines
      while ((line = r.readLine()) != null) {
        final String[] parts = line.trim().split(",");
        final String name = trimQuotes(parts[0]);
        if (columnName.equals(name)) {
          for (int i = 1; i < parts.length; ++i) {
            final String value = trimQuotes(parts[i]);
            if (!targetIndex.containsKey(value)) {
              targetIndex.put(value, targetIndex.size());
            }
            final String sampleName = sampleNames[i];
            nameIndex.put(sampleName, targetIndex.get(value));
          }
        }
      }
    }
    if (targetIndex.isEmpty()) {
      throw new IOException("Could not find field " + columnName);
    }
    final String[] names = getSampleNames();
    mClassIndexes = new int[names.length];
    for (int i = 0; i < names.length; ++i) {
      mClassIndexes[i] = nameIndex.getOrDefault(names[i], 0);
    }
    setSampleNames(names);
  }
}
