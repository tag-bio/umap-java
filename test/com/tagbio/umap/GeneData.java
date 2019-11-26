/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap;

import java.io.IOException;

public class GeneData extends Data {
  public GeneData() throws IOException {
    super("com/tagbio/umap/gene_exp_data.tsv.gz");
  }
}
