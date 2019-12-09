/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

// # Author: Leland McInnes <leland.mcinnes@gmail.com>
// # Enough simple sparse operations in numba to enable sparse UMAP
// #
// # License: BSD 3 clause
// from __future__ import print_function
// import numpy as np
// import numba

// from umap.utils import (
//     tau_rand_int,
//     tau_rand,
//     norm,
//     make_heap,
//     heap_push,
//     rejection_sample,
//     build_candidates,
//     deheap_sort,
// )

// import locale

// locale.setlocale(locale.LC_NUMERIC, "C")

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import tagbio.umap.metric.Metric;

final class Sparse {

  private Sparse() { }

// // Just reproduce a simpler version of numpy unique (not numba supported yet)
// @numba.njit()
// def arr_unique(arr):
//     aux = np.sort(arr)
//     flag = np.concatenate((np.ones(1, dtype=np.bool_), aux[1:] != aux[:-1]))
//     return aux[flag]

  private static int countDups(final int[] a) {
    int dups = 0;
    for (int k = 1; k < a.length; ++k) {
      if (a[k] == a[k - 1]) {
        ++dups;
      }
    }
    return dups;
  }


  static int[] arrUnique(final int[] a) {
    Arrays.sort(a);
    final int dups = countDups(a);
    if (dups == 0) {
      return a;
    }
    final int[] res = new int[a.length - dups];
    for (int k = 0, j = 0; k < a.length; ++k) {
      if (k == 0 || a[k] != a[k - 1]) {
        res[j++] = a[k];
      }
    }
    return res;
  }

  static int[] arrUnion(final int[] ar1, final int[] ar2) {
    if (ar1.length == 0) {
      return ar2;
    } else if (ar2.length == 0) {
      return ar1;
    } else {
      return arrUnique(MathUtils.concatenate(ar1, ar2));
    }
  }

  static int[] arrIntersect(final int[] ar1, final int[] ar2) {
   final int[] res = new int[Math.max(ar1.length, ar2.length)];
   int k = 0;
   int j = 0;
   int i = 0;
   while (j < ar1.length && i < ar2.length) {
     if (ar1[j] == ar2[i]) {
       res[k++] = ar1[j];
       ++j;
       ++i;
     } else if (ar1[j] < ar2[i]) {
       ++j;
     } else {
       ++i;
     }
   }
   return Arrays.copyOf(res, k);
 }


  static Object[] sparseSum(final int[] ind1, final float[] data1, final int[] ind2, final float[] data2) {
    final int[] resultInd = arrUnion(ind1, ind2);
    final float[] resultData = new float[resultInd.length];

    int i1 = 0;
    int i2 = 0;
    int nnz = 0;

    // pass through both index lists
    while (i1 < ind1.length && i2 < ind2.length) {
      final int j1 = ind1[i1];
      final int j2 = ind2[i2];

      if (j1 == j2) {
        final float val = data1[i1] + data2[i2];
        if (val != 0) {
          resultInd[nnz] = j1;
          resultData[nnz] = val;
          nnz += 1;
        }
        i1 += 1;
        i2 += 1;
      } else if (j1 < j2) {
        final float val = data1[i1];
        if (val != 0) {
          resultInd[nnz] = j1;
          resultData[nnz] = val;
          nnz += 1;
        }
        i1 += 1;
      } else {
        final float val = data2[i2];
        if (val != 0) {
          resultInd[nnz] = j2;
          resultData[nnz] = val;
          nnz += 1;
        }
        i2 += 1;
      }
    }

    // pass over the tails
    while (i1 < ind1.length) {
      final float val = data1[i1];
      if (val != 0) {
        resultInd[nnz] = i1;
        resultData[nnz] = val;
        nnz += 1;
      }
      i1 += 1;
    }

    while (i2 < ind2.length) {
      final float val = data2[i2];
      if (val != 0) {
        resultInd[nnz] = i2;
        resultData[nnz] = val;
        nnz += 1;
      }
      i2 += 1;
    }

    if (nnz == resultInd.length) {
      return new Object[] {resultInd, resultData};
    } else {
      // truncate to the correct length in case there were zeros created
      return new Object[]{Arrays.copyOf(resultInd, nnz), Arrays.copyOf(resultData, nnz)};
    }
  }


  static Object[] sparseDiff(final int[] ind1, final float[] data1, final int[] ind2, final float[] data2) {
    return sparseSum(ind1, data1, ind2, MathUtils.negate(data2));
  }


 static Object[] multiply(final int[] ind1, final float[] data1, final int[] ind2, final float[] data2) {
   final int[] resultInd = arrIntersect(ind1, ind2);
   final float[] resultData = new float[resultInd.length];

   int i1 = 0;
   int i2 = 0;
   int nnz = 0;

   // pass through both index lists
   while (i1 < ind1.length && i2 < ind2.length) {
     final int j1 = ind1[i1];
     final int j2 = ind2[i2];

     if (j1 == j2) {
       final float val = data1[i1] * data2[i2];
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
     return new Object[] {resultInd, resultData};
   } else {
     // truncate to the correct length in case there were zeros created
     return new Object[]{Arrays.copyOf(resultInd, nnz), Arrays.copyOf(resultData, nnz)};
   }
 }


// def make_sparse_nn_descent(sparse_dist, dist_args):
//     """Create a numba accelerated version of nearest neighbor descent
//     specialised for the given distance metric and metric arguments on sparse
//     matrix data provided in CSR ind, indptr and data format. Numba
//     doesn't support higher order functions directly, but we can instead JIT
//     compile the version of NN-descent for any given metric.

//     Parameters
//     ----------
//     sparse_dist: function
//         A numba JITd distance function which, given four arrays (two sets of
//         indices and data) computes a dissimilarity between them.

//     dist_args: tuple
//         Any extra arguments that need to be passed to the distance function
//         beyond the two arrays to be compared.

//     Returns
//     -------
//     A numba JITd function for nearest neighbor descent computation that is
//     specialised to the given metric.
//     """

//     @numba.njit(parallel=true)
//     def nn_descent(
//         inds,
//         indptr,
//         data,
//         n_vertices,
//         n_neighbors,
//         rng_state,
//         max_candidates=50,
//         n_iters=10,
//         delta=0.001,
//         rho=0.5,
//         rp_tree_init=true,
//         leaf_array=null,
//         verbose=false,
//     ):

//         current_graph = make_heap(n_vertices, n_neighbors)
//         for i in range(n_vertices):
//             indices = rejection_sample(n_neighbors, n_vertices, rng_state)
//             for j in range(indices.shape[0]):

//                 from_inds = inds[indptr[i] : indptr[i + 1]]
//                 from_data = data[indptr[i] : indptr[i + 1]]

//                 to_inds = inds[indptr[indices[j]] : indptr[indices[j] + 1]]
//                 to_data = data[indptr[indices[j]] : indptr[indices[j] + 1]]

//                 d = sparse_dist(from_inds, from_data, to_inds, to_data, *dist_args)

//                 heap_push(current_graph, i, d, indices[j], 1)
//                 heap_push(current_graph, indices[j], d, i, 1)

//         if rp_tree_init:
//             for n in range(leaf_array.shape[0]):
//                 for i in range(leaf_array.shape[1]):
//                     if leaf_array[n, i] < 0:
//                         break
//                     for j in range(i + 1, leaf_array.shape[1]):
//                         if leaf_array[n, j] < 0:
//                             break

//                         from_inds = inds[
//                             indptr[leaf_array[n, i]] : indptr[leaf_array[n, i] + 1]
//                         ]
//                         from_data = data[
//                             indptr[leaf_array[n, i]] : indptr[leaf_array[n, i] + 1]
//                         ]

//                         to_inds = inds[
//                             indptr[leaf_array[n, j]] : indptr[leaf_array[n, j] + 1]
//                         ]
//                         to_data = data[
//                             indptr[leaf_array[n, j]] : indptr[leaf_array[n, j] + 1]
//                         ]

//                         d = sparse_dist(
//                             from_inds, from_data, to_inds, to_data, *dist_args
//                         )

//                         heap_push(
//                             current_graph, leaf_array[n, i], d, leaf_array[n, j], 1
//                         )
//                         heap_push(
//                             current_graph, leaf_array[n, j], d, leaf_array[n, i], 1
//                         )

//         for n in range(n_iters):
//             if verbose:
//                 print("\t", n, " / ", n_iters)

//             candidate_neighbors = build_candidates(
//                 current_graph, n_vertices, n_neighbors, max_candidates, rng_state
//             )

//             c = 0
//             for i in range(n_vertices):
//                 for j in range(max_candidates):
//                     p = int(candidate_neighbors[0, i, j])
//                     if p < 0 || tau_rand(rng_state) < rho:
//                         continue
//                     for k in range(max_candidates):
//                         q = int(candidate_neighbors[0, i, k])
//                         if (
//                             q < 0
//                             || not candidate_neighbors[2, i, j]
//                             and not candidate_neighbors[2, i, k]
//                         ):
//                             continue

//                         from_inds = inds[indptr[p] : indptr[p + 1]]
//                         from_data = data[indptr[p] : indptr[p + 1]]

//                         to_inds = inds[indptr[q] : indptr[q + 1]]
//                         to_data = data[indptr[q] : indptr[q + 1]]

//                         d = sparse_dist(
//                             from_inds, from_data, to_inds, to_data, *dist_args
//                         )

//                         c += heap_push(current_graph, p, d, q, 1)
//                         c += heap_push(current_graph, q, d, p, 1)

//             if c <= delta * n_neighbors * n_vertices:
//                 break

//         return deheap_sort(current_graph)

//     return nn_descent


  static void generalSsetIntersection(final int[] indptr1, final int[] indices1, final float[] data1, final int[] indptr2, final int[] indices2, final float[] data2, final int[] resultRow, final int[] resultCol, final float[] resultVal, final float mixWeight) {

    final float leftMin = Math.max(MathUtils.min(data1) / 2.0F, 1.0e-8F);
    final float rightMin = Math.max(MathUtils.min(data2) / 2.0F, 1.0e-8F);

    for (int idx = 0; idx < resultRow.length; ++idx) {
      final int i = resultRow[idx];
      final int j = resultCol[idx];

      float leftVal = leftMin;
      for (int k = indptr1[i]; k < indptr1[i + 1]; ++k) {
        if (indices1[k] == j) {
          leftVal = data1[k];
        }
      }

      float rightVal = rightMin;
      for (int k = indptr2[i]; k < indptr2[i + 1]; ++k) {
        if (indices2[k] == j) {
          rightVal = data2[k];
        }
      }

      if (leftVal > leftMin || rightVal > rightMin) {
        if (mixWeight < 0.5) {
          resultVal[idx] = (float) (leftVal * Math.pow(rightVal, mixWeight / (1.0 - mixWeight)));
        } else {
          resultVal[idx] = (float) (Math.pow(leftVal, (1.0 - mixWeight) / mixWeight) * rightVal);
        }
      }
    }
  }




// @numba.njit()
// def sparse_euclidean(ind1, data1, ind2, data2):
//     aux_inds, aux_data = sparse_diff(ind1, data1, ind2, data2)
//     result = 0.0
//     for i in range(aux_data.shape[0]):
//         result += aux_data[i] ** 2
//     return np.sqrt(result)


// @numba.njit()
// def sparse_manhattan(ind1, data1, ind2, data2):
//     aux_inds, aux_data = sparse_diff(ind1, data1, ind2, data2)
//     result = 0.0
//     for i in range(aux_data.shape[0]):
//         result += np.abs(aux_data[i])
//     return result


// @numba.njit()
// def sparse_chebyshev(ind1, data1, ind2, data2):
//     aux_inds, aux_data = sparse_diff(ind1, data1, ind2, data2)
//     result = 0.0
//     for i in range(aux_data.shape[0]):
//         result = max(result, np.abs(aux_data[i]))
//     return result


// @numba.njit()
// def sparse_minkowski(ind1, data1, ind2, data2, p=2.0):
//     aux_inds, aux_data = sparse_diff(ind1, data1, ind2, data2)
//     result = 0.0
//     for i in range(aux_data.shape[0]):
//         result += np.abs(aux_data[i]) ** p
//     return result ** (1.0 / p)


// @numba.njit()
// def sparse_hamming(ind1, data1, ind2, data2, n_features):
//     num_not_equal = sparse_diff(ind1, data1, ind2, data2)[0].shape[0]
//     return float(num_not_equal) / n_features


// @numba.njit()
// def sparse_canberra(ind1, data1, ind2, data2):
//     abs_data1 = np.abs(data1)
//     abs_data2 = np.abs(data2)
//     denom_inds, denom_data = sparse_sum(ind1, abs_data1, ind2, abs_data2)
//     denom_data = 1.0 / denom_data
//     numer_inds, numer_data = sparse_diff(ind1, data1, ind2, data2)
//     numer_data = np.abs(numer_data)

//     val_inds, val_data = sparse_mul(numer_inds, numer_data, denom_inds, denom_data)

//     return np.sum(val_data)


// @numba.njit()
// def sparse_bray_curtis(ind1, data1, ind2, data2):  // pragma: no cover
//     abs_data1 = np.abs(data1)
//     abs_data2 = np.abs(data2)
//     denom_inds, denom_data = sparse_sum(ind1, abs_data1, ind2, abs_data2)

//     if denom_data.shape[0] == 0:
//         return 0.0

//     denominator = np.sum(denom_data)

//     numer_inds, numer_data = sparse_diff(ind1, data1, ind2, data2)
//     numer_data = np.abs(numer_data)

//     numerator = np.sum(numer_data)

//     return float(numerator) / denominator


// @numba.njit()
// def sparse_jaccard(ind1, data1, ind2, data2):
//     num_non_zero = arr_union(ind1, ind2).shape[0]
//     num_equal = arr_intersect(ind1, ind2).shape[0]

//     if num_non_zero == 0:
//         return 0.0
//     else:
//         return float(num_non_zero - num_equal) / num_non_zero


// @numba.njit()
// def sparse_matching(ind1, data1, ind2, data2, n_features):
//     num_true_true = arr_intersect(ind1, ind2).shape[0]
//     num_non_zero = arr_union(ind1, ind2).shape[0]
//     num_not_equal = num_non_zero - num_true_true

//     return float(num_not_equal) / n_features


// @numba.njit()
// def sparse_dice(ind1, data1, ind2, data2):
//     num_true_true = arr_intersect(ind1, ind2).shape[0]
//     num_non_zero = arr_union(ind1, ind2).shape[0]
//     num_not_equal = num_non_zero - num_true_true

//     if num_not_equal == 0.0:
//         return 0.0
//     else:
//         return num_not_equal / (2.0 * num_true_true + num_not_equal)


// @numba.njit()
// def sparse_kulsinski(ind1, data1, ind2, data2, n_features):
//     num_true_true = arr_intersect(ind1, ind2).shape[0]
//     num_non_zero = arr_union(ind1, ind2).shape[0]
//     num_not_equal = num_non_zero - num_true_true

//     if num_not_equal == 0:
//         return 0.0
//     else:
//         return float(num_not_equal - num_true_true + n_features) / (
//             num_not_equal + n_features
//         )


// @numba.njit()
// def sparse_rogers_tanimoto(ind1, data1, ind2, data2, n_features):
//     num_true_true = arr_intersect(ind1, ind2).shape[0]
//     num_non_zero = arr_union(ind1, ind2).shape[0]
//     num_not_equal = num_non_zero - num_true_true

//     return (2.0 * num_not_equal) / (n_features + num_not_equal)


// @numba.njit()
// def sparse_russellrao(ind1, data1, ind2, data2, n_features):
//     if ind1.shape[0] == ind2.shape[0] and np.all(ind1 == ind2):
//         return 0.0

//     num_true_true = arr_intersect(ind1, ind2).shape[0]

//     if num_true_true == np.sum(data1 != 0) and num_true_true == np.sum(data2 != 0):
//         return 0.0
//     else:
//         return float(n_features - num_true_true) / (n_features)


// @numba.njit()
// def sparse_sokal_michener(ind1, data1, ind2, data2, n_features):
//     num_true_true = arr_intersect(ind1, ind2).shape[0]
//     num_non_zero = arr_union(ind1, ind2).shape[0]
//     num_not_equal = num_non_zero - num_true_true

//     return (2.0 * num_not_equal) / (n_features + num_not_equal)


// @numba.njit()
// def sparse_sokal_sneath(ind1, data1, ind2, data2):
//     num_true_true = arr_intersect(ind1, ind2).shape[0]
//     num_non_zero = arr_union(ind1, ind2).shape[0]
//     num_not_equal = num_non_zero - num_true_true

//     if num_not_equal == 0.0:
//         return 0.0
//     else:
//         return num_not_equal / (0.5 * num_true_true + num_not_equal)


// @numba.njit()
// def sparse_cosine(ind1, data1, ind2, data2):
//     aux_inds, aux_data = sparse_mul(ind1, data1, ind2, data2)
//     result = 0.0
//     norm1 = norm(data1)
//     norm2 = norm(data2)

//     for i in range(aux_data.shape[0]):
//         result += aux_data[i]

//     if norm1 == 0.0 and norm2 == 0.0:
//         return 0.0
//     else if norm1 == 0.0 || norm2 == 0.0:
//         return 1.0
//     else:
//         return 1.0 - (result / (norm1 * norm2))


// @numba.njit()
// def sparse_correlation(ind1, data1, ind2, data2, n_features):

//     mu_x = 0.0
//     mu_y = 0.0
//     dot_product = 0.0

//     if ind1.shape[0] == 0 and ind2.shape[0] == 0:
//         return 0.0
//     else if ind1.shape[0] == 0 || ind2.shape[0] == 0:
//         return 1.0

//     for i in range(data1.shape[0]):
//         mu_x += data1[i]
//     for i in range(data2.shape[0]):
//         mu_y += data2[i]

//     mu_x /= n_features
//     mu_y /= n_features

//     shifted_data1 = np.empty(data1.shape[0], dtype=np.float32)
//     shifted_data2 = np.empty(data2.shape[0], dtype=np.float32)

//     for i in range(data1.shape[0]):
//         shifted_data1[i] = data1[i] - mu_x
//     for i in range(data2.shape[0]):
//         shifted_data2[i] = data2[i] - mu_y

//     norm1 = np.sqrt(
//         (norm(shifted_data1) ** 2) + (n_features - ind1.shape[0]) * (mu_x ** 2)
//     )
//     norm2 = np.sqrt(
//         (norm(shifted_data2) ** 2) + (n_features - ind2.shape[0]) * (mu_y ** 2)
//     )

//     dot_prod_inds, dot_prod_data = sparse_mul(ind1, shifted_data1, ind2, shifted_data2)

//     common_indices = set(dot_prod_inds)

//     for i in range(dot_prod_data.shape[0]):
//         dot_product += dot_prod_data[i]

//     for i in range(ind1.shape[0]):
//         if ind1[i] not in common_indices:
//             dot_product -= shifted_data1[i] * (mu_y)

//     for i in range(ind2.shape[0]):
//         if ind2[i] not in common_indices:
//             dot_product -= shifted_data2[i] * (mu_x)

//     all_indices = arr_union(ind1, ind2)
//     dot_product += mu_x * mu_y * (n_features - all_indices.shape[0])

//     if norm1 == 0.0 and norm2 == 0.0:
//         return 0.0
//     else if dot_product == 0.0:
//         return 1.0
//     else:
//         return 1.0 - (dot_product / (norm1 * norm2))


  static final Map<String, Metric> SPARSE_NAMED_DISTANCES = new HashMap<>();
  static {
    // todo
    SPARSE_NAMED_DISTANCES.put("euclidean", null);
//     // general minkowski distances
//     "euclidean": sparse_euclidean,
//     "manhattan": sparse_manhattan,
//     "l1": sparse_manhattan,
//     "taxicab": sparse_manhattan,
//     "chebyshev": sparse_chebyshev,
//     "linf": sparse_chebyshev,
//     "linfty": sparse_chebyshev,
//     "linfinity": sparse_chebyshev,
//     "minkowski": sparse_minkowski,
//     // Other distances
//     "canberra": sparse_canberra,
//     // 'braycurtis': sparse_bray_curtis,
//     // Binary distances
//     "hamming": sparse_hamming,
//     "jaccard": sparse_jaccard,
//     "dice": sparse_dice,
//     "matching": sparse_matching,
//     "kulsinski": sparse_kulsinski,
//     "rogerstanimoto": sparse_rogers_tanimoto,
//     "russellrao": sparse_russellrao,
//     "sokalmichener": sparse_sokal_michener,
//     "sokalsneath": sparse_sokal_sneath,
//     "cosine": sparse_cosine,
//     "correlation": sparse_correlation,
// }
  }


  static final Set<String> SPARSE_NEED_N_FEATURES = new HashSet<>();
  static {
    SPARSE_NEED_N_FEATURES.add("hamming");
    SPARSE_NEED_N_FEATURES.add("matching");
    SPARSE_NEED_N_FEATURES.add("kulsinski");
    SPARSE_NEED_N_FEATURES.add("rogerstanimoto");
    SPARSE_NEED_N_FEATURES.add("russellrao");
    SPARSE_NEED_N_FEATURES.add("sokalmichener");
    SPARSE_NEED_N_FEATURES.add("correlation");
  }
}
