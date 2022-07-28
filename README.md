# Cluster Validation based on Fisher's Linear Discriminant Analysis

This repository contains the implementation of a cluster validation approach to determine
the optimal number of clusters. For detailed information the reader is 

## Getting started
We use a [Conda](https://docs.conda.io/en/latest/miniconda.html) environment for dependency management. 
Therefore the minimal requirements are Conda. To create an environment, open an Anaconda prompt and run the following:
```
conda env create -f clustervalidation.yml
```

## Example

```python
# import library
from ClusterValidation import *

# further imports 
from sklearn.datasets import make_blobs

X, y = make_blobs(100, n_features=2, centers=3)

validator = CVFLDA(X, y).validate()

adjusted_labels = validator.adjusted_y
num_cl = validator.get_adjusted_cluster_count()

```

#### 

