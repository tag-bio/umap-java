package com.tagbio.umap;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

abstract class Data {
    private float[][] mData;
    private final List<String> mAttributes= new ArrayList<>();
    private final List<String> mSampleNames= new ArrayList<>();

    Data(String dataFile) {
        final ClassLoader classloader = Thread.currentThread().getContextClassLoader();
        final InputStream is = classloader.getResourceAsStream(dataFile);

        final List<List<String>> records = new ArrayList<>();

        try (final Scanner scanner = new Scanner(is)) {
            if (scanner.hasNextLine()) {
                // header line
                final String line = scanner.nextLine().trim();
                try (final Scanner rowScanner = new Scanner(line)) {
                    rowScanner.useDelimiter("\t");
                    if (rowScanner.hasNext()) {
                        assert("sample".equals(rowScanner.next()));
                    }
                    while (rowScanner.hasNext()) {
                        mAttributes.add(rowScanner.next());
                    }
                }
            }
            while (scanner.hasNextLine()) {
                final String line = scanner.nextLine().trim();
                final List<String> values = new ArrayList<>();
                try (final Scanner rowScanner = new Scanner(line)) {
                    rowScanner.useDelimiter("\t");
                    if (rowScanner.hasNext()) {
                        mSampleNames.add(rowScanner.next().trim());
                    }
                    while (rowScanner.hasNext()) {
                        values.add(rowScanner.next());
                    }
                }
                records.add(values);
             }
        }
        this.mData = new float[records.size()][records.get(0).size()];
        for (int j = 0; j < records.size(); j++) {
            final List<String> row = records.get(j);
            for (int i = 0; i < row.size(); i++) {
                this.mData[j][i] = Float.parseFloat(row.get(i));
            }
        }
    }

    public float[][] getData() {
        return this.mData;
    }

    public String[] getAttributes() {
        return mAttributes.toArray(new String[mAttributes.size()]);
    }

    public String[] getSampleNames() {
        return mSampleNames.toArray(new String[mSampleNames.size()]);
    }
}

