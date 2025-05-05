SConnect
=======
SConnect: Identifying the similar cell types and cell states across heterogeneous studies.


Tutorial
========
See `docs/SConnect_tutorial.ipynb` for a detailed tutorial on how to use SConnect.


Installation
============

To run SConnect, we provide a docker image that contains all the necessary dependencies: https://hub.docker.com/repository/docker/felix0097/sconnect/general

To run the R differential expression testing code (pre-processing), we provide a separate docker image: https://hub.docker.com/repository/docker/felix0097/pseudobulk/general


Project structure
=================
* `datasim/`: Contains the code for the SConnect package and the differential expression testing code
* `docs/`: Contains the documentation and tutorial notebooks for SConnect
* `notebooks/`
  * `DEG-filtering/`: Code used to create gene list to filter differentially expressed genes.
  * `evaluation/`: Analysis code for paper
  * `example-data/`: Code used to download and preprocess the data used in the paper
  * `similarity-methods/`: `CellHint.ipynb` and `pyMN.ipynb` contains code to fit CellHint and MetaNeighbor reference models
* `scripts/`: Contains the scripts used to train SConnect


Licence
-------
MIT license


System requirements
===================
Operating system: Ubuntu 22.04 LTS (used OS version)

Python version: 3.10 or higher

Packages: 
* See pyproject.toml for a list of required packages
* See `datasim/de_testing/testing/INSTALL_R_PACKAGES` for a list of required R packages


Hardware requirements
=====================
To scale to atlas-scale datasets, we recommend:
* We recommend a modern GPU with at least 40GB of VRAM
* Moreover, for the pre-processing code, we recommend around 200GB of system memory


Authors
-------
`SConnect` was written by `Felix Fischer <felix.fischer@helmholtz-munich.de>`

Support for software development, testing, modeling, and benchmarking provided by the Cell Annotation Platform team 
(Roman Mukhin, Andrey Isaev, Uğur Bayındır)
