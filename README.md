Java UMAP
=========

A self-contained native Java implementation of UMAP based on the
reference [Python implementation](https://github.com/lmcinnes/umap).

This implementation has been designed and developed by Tag.bio in conjunction with Real Time Genomics.
https://tag.bio/
https://www.realtimegenomics.com/

Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction
technique that can be used for visualization similarly to t-SNE, but also for
general non-linear dimension reduction. The algorithm is founded on three
assumptions about the data:

1. The data is uniformly distributed on a Riemannian manifold;
2. The Riemannian metric is locally constant (or can be approximated as such);
3. The manifold is locally connected.

From these assumptions it is possible to model the manifold with a fuzzy
topological structure. The embedding is found by searching for a low dimensional
projection of the data that has the closest possible equivalent fuzzy
topological structure.

The details for the underlying mathematics and algorithms can be found in
L. McInnes and J. Healy, J. Melville,
[UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426).

How to use UMAP
---------------

Using Java UMAP is simple:

```java
final float[][] data = ...           // input data instances * attributes
final Umap umap = new Umap();
umap.setNumberComponents(2);         // number of dimensions in result
umap.setNumberNearestNeighbours(15);
umap.setThreads(1);                  // use > 1 to enable parallelism
final float[][] result = umap.fitTransform(data);
```

There are a large number of potential parameters than can be set; the
major ones are as follows:

*  `setNumberNearestNeighbours`: This determines the number of neighboring points used in
   local approximations of manifold structure. Larger values will result in
   more global structure being preserved at the loss of detailed local
   structure. In general this parameter should often be in the range 5 to
   50, with a choice of 10 to 15 being a sensible default.

*  `setMinDist`: This controls how tightly the embedding is allowed compress
   points together. Larger values ensure embedded points are more evenly
   distributed, while smaller values allow the algorithm to optimize more
   accurately with regard to local structure. Sensible values are in the
   range 0.001 to 0.5, with 0.1 being a reasonable default.

*  `setMetric`: This determines the choice of metric used to measure distance
   in the input space. Default to a Euclidean metric.

In addition the number of threads to use can be set with `setThreads`.  If this is
set to a number greater than 1, then the results will no longer be deterministic,
even for a specified random number seed.

Limitations
-----------

This Java implementation has a number of limitations when compared to the reference
Python implementation:

* Only the `random` initialization mode is currently supported.  In particular,
  `spectral` initialization is not currently supported.

* The `transform()` method for adding new points to an existing embedding is
  implemented, but should be considered alpha.

* Selection of curve parameters is more limited than in the Python version
  (an `IllegalArgumentException` will be reported if limits are exceeded).

* Other limitations might occur as an `UnsupportedOperationException`.

Citation
--------

If you would like to cite this algorithm in your work the ArXiv paper is the
current reference:

```bibtex

   @article{2018arXivUMAP,
        author = {{McInnes}, L. and {Healy}, J. and {Melville}, J.},
        title = "{UMAP: Uniform Manifold Approximation
        and Projection for Dimension Reduction}",
        journal = {ArXiv e-prints},
        archivePrefix = "arXiv",
        eprint = {1802.03426},
        primaryClass = "stat.ML",
        keywords = {Statistics - Machine Learning,
                    Computer Science - Computational Geometry,
                    Computer Science - Learning},
        year = 2018,
        month = feb,
   }
```

License
-------

The Java UMAP package is 3-clause BSD licensed.

