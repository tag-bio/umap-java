/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap;

import java.io.IOException;

public class DigitData extends Data {
  public DigitData() throws IOException {
    super("com/tagbio/umap/digits.tsv");
  }

  @Override
  String getName() {
    return "digit";
  }
}
