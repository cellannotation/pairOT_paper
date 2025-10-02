[//]: # (# pairOT)

<h1 align="center">
  <img src="docs/pairOT-logo.png" alt="pairOT Logo" width="500"/>
</h1>

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


- [Introduction](#pairOT-identifying-the-similar-cell-types-and-cell-states-across-heterogeneous-studies)
- [Basics of using pairOT](#basics-of-using-pairOT)
- [Tutorial](#tutorial)
- [Installation](#installation)
  - [Running pairOT via Docker](#running-pairOT-via-docker)
  - [Installing pairOT via pip](#install-pairOT-via-pip)
- [Project structure](#project-structure)
- [References](#references)

## pairOT: Identifying the similar cell types and cell states across heterogeneous studies.

* **What is pairOT?**
  
  pairOT is a Python package to align cell annotations between single-cell RNA-seq datasets. Given the cell-type 
annotations of two datasets, pairOT shows the similarity of the respective cell type clusters. Hence, pairOT can be 
used to identify similar cell types and cell states between the two studies and to show to the user where there might be 
disagreements between cell annotations of the two datasets.


* **How does it work?** ![pairOT](docs/pairOT.png)
pairOT aims to “align” (or to “connect”) cell-type labels between two scRNA-seq datasets 
(query + reference dataset), suggesting similar clusters or potential matches against the reference dataset for each 
cell type in the query dataset solely based on the underlying transcriptomic signatures. To achieve this task, we model 
each dataset as a point cloud, meaning each data point or cell is associated with a gene expression vector (𝘅) and a 
cluster / cell-type label (y). Notably, the clustering or grouping information of individual cells uniformly annotated, 
not the associated string labels, is provided by the cell-type label y. At its core, pairOT uses optimal transport to 
connect the point cloud distribution of the query dataset with the point cloud from the reference dataset; thus, the 
method considers the global structure of the data, compared to just finding the closest neighbors in the reference dataset.  
The distance between a cell in the query dataset (described by the multi-dimensional gene expression vector 𝘅₁ and 
cluster label y₁) and a cell in the reference dataset (described by the multi-dimensional gene expression vector 
𝘅₂ and cluster label y₂) is split into two parts. The distance measure (also known as the “transport cost”) is the 
sum between the distance in gene expression space and the distance in label space:
![distance_equation](docs/distance_equation.png)

* **What does it produce?**

  pairOT produces two key outputs:

  * **Similarity Matrix**: Showing the closest matching (considering the global structure of the data) cell-type in the 
reference dataset for each cell-type in the query dataset.
![mapping](docs/mapping_example.png)
  * **Distance Matrix**: Showing the similarity/distance between the cell-type clusters of the query and reference 
dataset.
![distance](docs/distance_example.png)

* **What are the advantages over existing methods?** 

  We improve over existing methods in two key ways:
  1. Using a considerably more predictive distance measure: By including information of differentially expressed genes 
in the distance measure we're able to resolve differences between fine-grained cell-type subtypes.
  2. Considering the global structure of the data: By using optimal transport we consider the global structure of the 
data and thus are able to provide more trustworthy suggestions.


## Basics of using pairOT

```python
import scanpy as sc

from pairot.preprocessing import preprocess_adatas

# 1. Preprocess input data
adata_query, adata_ref = preprocess_adatas(
    sc.read_h5ad("path/to/query.h5ad"),
    sc.read_h5ad("path/to/reference.h5ad"),
    n_top_genes=750,
    cell_type_column_adata1="cell_type_column_query",
    cell_type_column_adata2="cell_type_column_ref",
    sample_column_adata1="sequencing_sample_column_query",
    sample_column_adata2="sequencing_sample_column_ref",
)

# 2. Initialize pairOT model
from pairot.dataset_ot import DatasetMapping

dataset_map = DatasetMapping(adata_query, adata_ref)
dataset_map.init_geom(batch_size=512, epsilon=0.05)
dataset_map.init_problem(tau_a=1.0, tau_b=1.0)

# 3. Fit pairOT model
dataset_map.solve()
mapping = dataset_map.compute_cluster_mapping(aggregation_method="mean")
distance = dataset_map.compute_cluster_distances()

# 4. Visualize results
from pairot.plotting import plot_cluster_mapping, plot_cluster_distance

plot_cluster_mapping(mapping)  # similarity matrix
distance = distance.loc[
    mapping.max(axis=1).sort_values(ascending=False).index.tolist(),
    mapping.max().sort_values(ascending=False).index.tolist(),
]  # order cluster distance matrix the same way as similarity matrix
plot_cluster_distance(distance)  # cluster distance matrix
```


## Tutorial
Please see the following tutorials for detailed examples of how to use pairOT:

### pairOT: Detailed explanation
This tutorial gives detailed instructions on how to use pairOT.
* [Jupyter Notebook](https://github.com/cellannotation/dataset-similarity/blob/devel/docs/pairOT_tutorial.ipynb)
* [HTML version](https://htmlpreview.github.io/?https://raw.githubusercontent.com/cellannotation/pairOT/refs/heads/devel/docs/pairOT_tutorial.html)

### pairOT: Fit pairOT with reduced compute requirements / Speed up pairOT computations
This tutorial shows how to speedup pairOT computations. This is especially relevant if only limited compute resources
are available or for very large datasets.
* [Jupyter Notebook](https://github.com/cellannotation/dataset-similarity/blob/devel/docs/pairOT_tutorial_reduce_compute_requirements.ipynb)
* [HTML version](https://htmlpreview.github.io/?https://raw.githubusercontent.com/cellannotation/pairOT/refs/heads/devel/docs/pairOT_tutorial_reduce_compute_requirements.html)

## Installation

### Running pairOT via Docker
To run pairOT, we provide a docker image that contains all the necessary dependencies: https://hub.docker.com/r/felix0097/pairot/tags
```bash
docker pull felix0097/pairot:full_v1
```

### Install pairOT via pip

#### Install R
To run the R differential expression testing code (pre-processing), you'll need to install R on your system.

You can either do it via Anaconda:
```bash
conda install conda-forge::r-base
```
Or install R directly on your system. 
Please refer to the official R documentation for installation instructions: https://cran.r-project.org/bin/linux/ubuntu/fullREADME.html#installing-r
#### Install pairOT via pip
```bash
git clone git@github.com:cellannotation/pairOT.git
cd pairOT
pip install ".[de_testing]"
```

By default, the installed JAX version only uses the CPU to make JAX recognize your GPU/TPU, 
see https://docs.jax.dev/en/latest/installation.html#installation
```bash
pip install -U "jax[cuda12]"
```

#### Install R dependencies
To install the required R dependencies, open a Python console and run the following commands:

```python
import rpy2.robjects as ro

INSTALL_R_PACKAGES_LATETST = """
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
    
BiocManager::install("limma")
BiocManager::install("rhdf5")
install.packages("Matrix")
install.packages("magrittr")
install.packages("data.table")
install.packages("glue")
install.packages("stringr")
"""

ro.r(INSTALL_R_PACKAGES_LATETST)
```
It might take a while to install all R dependencies.


### System requirements
Operating system: Ubuntu 22.04 LTS (used OS version)

Python version: 3.10 or higher


### Hardware requirements
To scale to atlas-scale datasets, we recommend:
* We recommend a modern GPU with at least 40GB of VRAM
* Moreover, for the pre-processing code, we recommend around 128GB of system memory (depending on dataset size)


## Project structure
* `pairot/`: Contains the code for the pairOT package and the differential expression testing code
* `docs/`: Contains the documentation and tutorial notebooks for pairOT
* `notebooks/`
  * `DEG-filtering/`: Code used to create gene list to filter differentially expressed genes.
  * `evaluation/`: Analysis code for paper
  * `example-data/`: Code used to download and preprocess the data used in the paper
  * `similarity-methods/`: `CellHint.ipynb` and `pyMN.ipynb` contains code to fit CellHint and MetaNeighbor reference models
* `scripts/`: Contains the scripts used to train pairOT


## Licence
MIT license


## References
`pairOT` was written by `Felix Fischer <felix.fischer@helmholtz-munich.de>`

Support for software development, testing, modeling, and benchmarking provided by the Cell Annotation Platform team 
(Roman Mukhin)
