/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.util.Arrays;

/**
 * Vector.
 * @author Sean A. Irvine
 * @author Richard Littin
 */
class SparseVector {

  final int[] mIndices;
  final float[] mData;

  /**
   * Vector.
   * @param indices indices of nonzero elements
   * @param data nonzero elements
   */
  SparseVector(final int[] indices, final float[] data) {
    mIndices = indices;
    mData = data;
  }

  float norm() {
    return Utils.norm(mData);
  }

  void divide(final float v) {
    for (int k = 0; k < mData.length; ++k) {
      mData[k] /= v;
    }
  }

  SparseVector negate() {
    final float[] neg = new float[mData.length];
    for (int k = 0; k < neg.length; ++k) {
      neg[k] = -mData[k];
    }
    return new SparseVector(mIndices, neg);
  }

  SparseVector add(final SparseVector right) {
    final int[] resultInd = Sparse.arrUnion(mIndices, right.mIndices);
    final float[] resultData = new float[resultInd.length];

    int i1 = 0;
    int i2 = 0;
    int nnz = 0;

    // pass through both index lists
    while (i1 < mIndices.length && i2 < right.mIndices.length) {
      final int j1 = mIndices[i1];
      final int j2 = right.mIndices[i2];

      if (j1 == j2) {
        final float val = mData[i1] + right.mData[i2];
        if (val != 0) {
          resultInd[nnz] = j1;
          resultData[nnz] = val;
          nnz += 1;
        }
        i1 += 1;
        i2 += 1;
      } else if (j1 < j2) {
        final float val = mData[i1];
        if (val != 0) {
          resultInd[nnz] = j1;
          resultData[nnz] = val;
          nnz += 1;
        }
        i1 += 1;
      } else {
        final float val = right.mData[i2];
        if (val != 0) {
          resultInd[nnz] = j2;
          resultData[nnz] = val;
          nnz += 1;
        }
        i2 += 1;
      }
    }

    // pass over the tails
    while (i1 < mIndices.length) {
      final float val = mData[i1];
      if (val != 0) {
        resultInd[nnz] = i1;
        resultData[nnz] = val;
        nnz += 1;
      }
      i1 += 1;
    }

    while (i2 < right.mIndices.length) {
      final float val = right.mData[i2];
      if (val != 0) {
        resultInd[nnz] = i2;
        resultData[nnz] = val;
        nnz += 1;
      }
      i2 += 1;
    }

    if (nnz == resultInd.length) {
      return new SparseVector(resultInd, resultData);
    } else {
      // truncate to the correct length in case there were zeros created
      return new SparseVector(Arrays.copyOf(resultInd, nnz), Arrays.copyOf(resultData, nnz));
    }
  }

  SparseVector multiply(SparseVector right) {
    final int[] resultInd = Sparse.arrIntersect(mIndices, right.mIndices);
    final float[] resultData = new float[resultInd.length];

    int i1 = 0;
    int i2 = 0;
    int nnz = 0;

    // pass through both index lists
    while (i1 < mIndices.length && i2 < right.mIndices.length) {
      final int j1 = mIndices[i1];
      final int j2 = right.mIndices[i2];

      if (j1 == j2) {
        final float val = mData[i1] * right.mData[i2];
        if (val != 0) {
          resultInd[nnz] = j1;
          resultData[nnz] = val;
          ++nnz;
        }
        ++i1;
        ++i2;
      } else if (j1 < j2) {
        ++i1;
      } else {
        ++i2;
      }
    }

    if (nnz == resultInd.length) {
      return new SparseVector(resultInd, resultData);
    } else {
      // truncate to the correct length in case there were zeros created
      return new SparseVector(Arrays.copyOf(resultInd, nnz), Arrays.copyOf(resultData, nnz));
    }
  }

  float sum() {
    float sum = 0;
    for (final float v : mData) {
      sum += v;
    }
    return sum;
  }

}
