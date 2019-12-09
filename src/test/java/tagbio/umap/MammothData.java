/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.io.IOException;

public class MammothData extends Data {
  public MammothData() throws IOException {
    super("tagbio/umap/mammoth.tsv");
  }

  @Override
  String getName() {
    return "mammoth";
  }
}
