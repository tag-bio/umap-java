package com.tagbio.umap;

public class IrisData extends Data {
    public IrisData() {
        super("com/tagbio/umap/iris.tsv");
    }

    public static void main(String[] args) {
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
