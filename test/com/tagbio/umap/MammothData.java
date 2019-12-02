/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap;

import java.io.IOException;

public class MammothData extends Data {
  public MammothData() throws IOException {
    super("com/tagbio/umap/mammoth.tsv");
  }
}
