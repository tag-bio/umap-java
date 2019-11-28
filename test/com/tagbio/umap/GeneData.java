/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.util.HashMap;
import java.util.Map;

public class GeneData extends Data {
  public GeneData() throws IOException {
    super("com/tagbio/umap/gene_exp_data.tsv.gz");
  }

  public void setSampleNamesFromInfo(String columnName) throws IOException {
    final Map<String, Integer> targetIndex = new HashMap<>();

    try (final LineNumberReader r = new LineNumberReader(new InputStreamReader(getStream("com/tagbio/umap/gene_exp_info.tsv")))) {
      String line = r.readLine();
      if (line == null) {
        throw new IOException("No header line");
      }
      // Process header line
      line = line.trim();
      int columnNumber = -1;
      final String[] columns = line.trim().split("\t");
      for (int c = 0; c < columns.length; ++c) {
        if (columnName.equals(columns[c])) {
          columnNumber = c;
        }
      }
      if (columnNumber < 0) {
        throw new IllegalArgumentException("Could not find '" + columnName + "' in column names: " + line);
      }

      // Process data lines
      while ((line = r.readLine()) != null) {
        final String[] parts = line.trim().split("\t");
        final String value = parts[columnNumber];
        if (!targetIndex.containsKey(value)) {
          targetIndex.put(value, targetIndex.size());
        }
      }
    }

    final String[] attributes = getAttributes();
    for (int i = 0; i < attributes.length; ++i) {
      attributes[i] = targetIndex.getOrDefault(attributes[i], Integer.valueOf(0)).toString();
    }
    setAttributes(attributes);
  }
}
