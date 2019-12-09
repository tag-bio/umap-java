/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package com.tagbio.umap;

import java.io.IOException;

public class IrisData extends Data {
  public IrisData() throws IOException {
    super("com/tagbio/umap/iris.tsv");
  }

  public IrisData(boolean small) throws IOException {
    super(small ? "com/tagbio/umap/iris-small.tsv" : "com/tagbio/umap/iris.tsv");
  }

  @Override
  String getName() {
    return "iris";
  }

  public static void main(String[] args) throws IOException {
    Data id = new IrisData();

    System.out.println("Attributes:");
    for (String att : id.getAttributes()) {
      System.out.print(" " + att);
    }
    System.out.println();

    System.out.println("Sample Names:");
    for (String name : id.getSampleNames()) {
      System.out.println(name);
    }
  }
}
