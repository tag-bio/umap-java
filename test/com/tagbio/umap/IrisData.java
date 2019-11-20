package com.tagbio.umap;

public class IrisData extends Data {
    public IrisData() {
        super("com/tagbio/umap/iris.csv");
    }


    public static void main(String[] args) {
        Data id = new IrisData();

        System.out.println("Targets:");
        for (int target : id.getTargets()) {
            System.out.print(" " + target);
        }
        System.out.println();

        System.out.println("TargetNames:");
        for (String name : id.getTargetNames()) {
            System.out.println(name);
        }
    }
}
