package com.tagbio.umap;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

abstract class Data {
    private float[][] data;
    private int[] targets;
    private String[] targetNames;

    Data(String dataFile) {
        final ClassLoader classloader = Thread.currentThread().getContextClassLoader();
        final InputStream is = classloader.getResourceAsStream(dataFile);
        final List<List<String>> records = new ArrayList<>();
        String section = null;
        try (final Scanner scanner = new Scanner(is)) {
            while (scanner.hasNextLine()) {
                final String line = scanner.nextLine().trim();
                if (line.startsWith("[")) {
                    saveSection(section, records);
                    section = line;
                    records.clear();
                } else {
                    final List<String> values = new ArrayList<>();
                    try (final Scanner rowScanner = new Scanner(line)) {
                        rowScanner.useDelimiter(",");
                        while (rowScanner.hasNext()) {
                            values.add(rowScanner.next());
                        }
                    }
                    records.add(values);
                }
            }
        }
        saveSection(section, records);
    }

    private void saveSection(final String section, final List<List<String>> records) {
        if (section != null) {
            if ("[data]".equals(section)) {
                // convert to float[][]
                this.data = new float[records.size()][records.get(0).size()];
                for (int j = 0; j < records.size(); j++) {
                    final List<String> row = records.get(j);
                    for (int i = 0; i < row.size(); i++) {
                        this.data[j][i] = Float.parseFloat(row.get(i));
                    }
                }
            } else if ("[targets]".equals(section)) {
                // convert to int[]
                final List<String> row = records.get(0);
                this.targets = new int[row.size()];
                for (int i = 0; i < row.size(); i++) {
                    this.targets[i] = Integer.parseInt(row.get(i));
                }
            } else if ("[targetNames]".equals(section)) {
                // convert to String[]
                final int length = records.get(0).size();
                this.targetNames = records.get(0).toArray(new String[length]);
            } else {
                throw new IllegalArgumentException("Bad section " + section);
            }
        }
    }

    public float[][] getData() {
        return this.data;
    }

    public int[] getTargets() {
        return this.targets;
    }

    public String[] getTargetNames() {
        return this.targetNames;
    }
}

