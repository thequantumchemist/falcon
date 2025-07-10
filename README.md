# Falcon
This work presents the FALCON (Fast Active Learning for Computational ab initio mOlecular dyNamics) calculator where the ML model is trained on-the-fly (OTF) and uses its own uncertainty estimation to decide whether an exact calculation is required. 

![Alt text](FALCON_MD.png?raw=true "FALCON")

The FALCON calculator can be used with any machine learning model. A base class is provided in order to make different ML frameworks work together with FALCON.
However, as standart the Gaussian Process Regression (GPR) and its sparsified versions (SparseGPR) as implemented in the AGOX framework by Hammer and co-workers is implemented as default ML models.


# Authors
Noah Felis
Wilke Dononelli

------------

# Requirements
* Python_ 3.8 or later
* NumPy_ (base N-dimensional array package)
* ase_ 3.23 (functions to determine atomic structures' geometries and quantum chemical calculators)
* agox (Atomistic Global Optimization X)

------------

# Installation
FALCON can be installed by cloning the Git repository and installing it using `pip`:

```bash
git clone https://github.com/thequantumchemist/falcon
cd falcon
pip install .
```

Alternatively, you can add ``~/falcon_md`` to your $PYTHONPATH environment variable after cloning the repository.

------------

# Tutorial
A tutorial is available here: https://thequantumchemist.github.io/falcon/

------------

# Citation
When using FALCON, please cite the following papers:
