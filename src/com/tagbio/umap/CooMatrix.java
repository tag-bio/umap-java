package com.tagbio.umap;

import java.util.Arrays;

/**
 * @author Sean A. Irvine
 */
class CooMatrix extends Matrix {
  // todo -- replacement fo scipy coo_matrix

  // todo I think this is internal rep of data for this form of matrix -- currently some direct external access
  int[] row;
  int[] col;
  float[] data;

  CooMatrix(final float[] vals, final int[] rows, final int[] cols, final int[] lengths) {
    super(lengths);
    if (rows.length != cols.length || rows.length != vals.length) {
      throw new IllegalArgumentException();
    }
    row = rows;
    col = cols;
    data = vals;
    sort(0, rows.length);
    checkDataValid();
  }

  private void checkDataValid() {
    for (int r : row) {
      if (r < 0 || r >= shape[0]) {
        throw new IllegalArgumentException("Row index out of bounds: 0 <= " + r + " < " + shape[0]);
      }
    }
    for (int c : col) {
      if (c < 0 || c >= shape[1]) {
        throw new IllegalArgumentException("Column index out of bounds: 0 <= " + c + " < " + shape[1]);
      }
    }
    for (int i = 1; i < row.length; ++i) {
      if (compare(i, i - 1) == 0) {
        throw new IllegalArgumentException("Duplicated array position: row " + row[i] + ", col " + col[i]);
      }
    }
  }

  private void swap(final int a, final int b) {
      final int t = row[a];
      row[a] = row[b];
      row[b] = t;
      final int u = col[a];
      col[a] = col[b];
      col[b] = u;
      final float v = data[a];
      data[a] = data[b];
      data[b] = v;
  }

  private int compare(final int i, final int j) {
    int res = Integer.compare(row[i], row[j]);
    if (res == 0) {
      res = Integer.compare(col[i], col[j]);
    }
    return res;
  }

  private void sort(final int off, final int len) {
    // Insertion sort on smallest arrays
    if (len < 7) {
      for (int i = off; i < len + off; ++i) {
        for (int j = i; j > off && compare(j - 1, j) > 0; --j) {
          swap(j, j - 1);
        }
      }
      return;
    }

    // Choose a partition element, v
    int m = off + (len >> 1); // Small arrays, middle element
    if (len != 7) {
      int l = off;
      int n = off + len - 1;
      if (len > 40) { // Big arrays, pseudomedian of 9
        final int s = len / 8;
        l = med3(l, l + s, l + 2 * s);
        m = med3(m - s, m, m + s);
        n = med3(n - 2 * s, n - s, n);
      }
      m = med3(l, m, n); // Mid-size, med of 3
    }
    //final double v = x[m];

    // Establish Invariant: v* (<v)* (>v)* v*
    int a = off;
    int b = a;
    int c = off + len - 1;

    // Establish Invariant: v* (<v)* (>v)* v*
    int d = c;
    while (true) {
      while (b <= c && compare(b, m) <= 0) {
        if (compare(b, m) >= 0) {
          swap(a++, b);
        }
        ++b;
      }
      while (c >= b && compare(c, m) >= 0) {
        if (compare(c, m) <= 0) {
          swap(c, d--);
        }
        --c;
      }
      if (b > c) {
        break;
      }
      swap(b++, c--);
    }

    // Swap partition elements back to middle
    int s2;

    // Swap partition elements back to middle
    final int n2 = off + len;
    s2 = Math.min(a - off, b - a);
    vecswap(off, b - s2, s2);
    s2 = Math.min(d - c, n2 - d - 1);
    vecswap(b, n2 - s2, s2);

    // Recursively sort non-partition-elements
    if ((s2 = b - a) > 1) {
      sort(off, s2);
    }
    if ((s2 = d - c) > 1) {
      sort(n2 - s2, s2);
    }
  }

  private void vecswap(final int aa, final int bb, final int n) {
    for (int i = 0, a = aa, b = bb; i < n; ++i, ++a, ++b) {
      swap(a, b);
    }
  }

  int med3(final int a, final int b, final int c) {
    final int ab = compare(a, b);
    final int ac = compare(a, c);
    final int bc = compare(b, c);

    return ab < 0
      ? (bc < 0 ? b : ac < 0 ? c : a)
      : (bc > 0 ? b : ac > 0 ? c : a);
  }


  void sum_duplicates() {
    // todo add identical entries -- this would be fairly easy if we knew arrays we sorted by (row,col)
    // todo for now ugliness ...

    final DefaultMatrix res = new DefaultMatrix(shape);
    for (int k = 0; k < data.length; ++k) {
      res.set(row[k], col[k], res.get(row[k], col[k]) + data[k]);
    }
    CooMatrix coo = res.tocoo(); // todo yikes!!
    row = coo.row;
    col = coo.col;
    data = coo.data;
  }

  @Override
  float get(final int r, final int c) {
    // todo this could be made faster if can assume sorted row[] col[]
    // todo there may be duplicate coords, need to sum result? (or doc this problem away!)
    int left = 0;
    int right = row.length - 1;
    while (left <= right) {
      int mid = left + (right - left) / 2;
      // Check if x is present at mid
      if (row[mid] == r) {
        if (col[mid] == c) {
          return data[mid];
        }
        if (col[mid] < c) {
          left = mid + 1;
          // If x is smaller, ignore right half
        } else {
          right = mid - 1;
        }
      } else {
        // If x greater, ignore left half
        if (row[mid] < r) {
          left = mid + 1;
          // If x is smaller, ignore right half
        } else {
          right = mid - 1;
        }
      }
    }
    return 0;
  }

  @Override
  void set(final int row, final int col, final float val) {
    throw new UnsupportedOperationException();
  }

  @Override
  Matrix copy() {
    System.out.println("coo copy");
    return new CooMatrix(Arrays.copyOf(data, data.length), Arrays.copyOf(row, row.length), Arrays.copyOf(col, col.length), Arrays.copyOf(shape, shape.length));
  }

  @Override
  Matrix transpose() {
    // todo note this is not copying the arrays -- might be a mutability issue
    //return new CooMatrix(data, col, row, new int[] {shape[1], shape[0]});
    return super.transpose().tocoo();
  }

  @Override
  CooMatrix tocoo() {
    return this;
  }

  void eliminate_zeros() {
    int zeros = 0;
    for (final float v : data) {
      if (v == 0) {
        ++zeros;
      }
    }
    if (zeros > 0) {
      final int[] r = new int[row.length - zeros];
      final int[] c = new int[row.length - zeros];
      final float[] d = new float[row.length - zeros];
      for (int k = 0, j = 0; k < data.length; ++k) {
        if (data[k] != 0) {
          r[j] = row[k];
          c[j] = col[k];
          d[j++] = data[k];
        }
      }
      row = r;
      col = c;
      data = d;
    }
  }

  @Override
  Matrix add(final Matrix m) {
    // todo this could do this without using super
    return super.add(m).tocoo();
  }

  @Override
  Matrix subtract(final Matrix m) {
    // todo this could do this without using super
    return super.subtract(m).tocoo();
  }

  @Override
  Matrix pointwiseMultiply(final Matrix m) {
    // todo this could do this without using super
    return super.pointwiseMultiply(m).tocoo();
  }

  @Override
  Matrix multiply(final Matrix m) {
    if (!(m instanceof CooMatrix)) {
      return super.multiply(m).tocoo();
    }
    // We are multiplying two CooMatrices together
    // todo can this be made faster?
    final CooMatrix a = (CooMatrix) m;
    if (shape[1] != m.shape[0]) {
      throw new IllegalArgumentException("Incompatible matrix sizes");
    }
    final int rows = shape[0];
    final int cols = m.shape[1];
    final float[][] res = new float[rows][cols];
    for (int k = 0; k < data.length; ++k) {
      final int r = row[k];
      final int c = col[k];
      for (int j = 0; j < a.data.length; ++j) {
        if (a.row[j] == c) {
          res[r][a.col[j]] += data[k] * a.data[j];
        }
      }
    }
    return new DefaultMatrix(res).tocoo();
  }

  @Override
  Matrix multiply(final float x) {
    final float[] newData = Arrays.copyOf(data, data.length);
    for (int i = 0; i < newData.length; ++i) {
      newData[i] *= x;
    }
    return new CooMatrix(newData, row, col, shape);
  }

  String sparseToString() {
    final StringBuilder sb = new StringBuilder();
    for (int k = 0; k < data.length; ++k) {
      sb.append('(').append(row[k]).append(", ").append(col[k]).append(") ").append(data[k]).append('\n');
    }
    return sb.toString();
  }
}
