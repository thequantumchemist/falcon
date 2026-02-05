# FALCON
This work presents the **FALCON** (**F**ast **A**ctive **L**earning for **C**omputational ab initio m**O**lecular dy**N**amics) calculator where the ML model is trained on-the-fly (OTF) and uses its own uncertainty estimation to decide whether an exact calculation is required. 

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
FALCON can be installed by installing it using `pip`:

```bash
pip install falcon-md
```



Alternatively, you can clone the Git repository:

```bash
git clone https://github.com/thequantumchemist/falcon
```
and add ``~/falcon_md`` to your $PYTHONPATH environment variable.

------------

# Tutorial

The tutorial directory of this repository contains three example scripts demonstrating how to use **FALCON** and introduce its main concepts:.
1. Basic OTF molecular dynamics with a default ML model.
2. Advanced OTF training with a customized Sparse Gaussian Process model.
3. Postprocessing and analysis of simulation results.

A detailed explanation is given in the ``README.md`` in the tutorial directory.

------------

# Citation
When using FALCON, please cite:

Felis, N., Dononelli, W. (2025). FALCON: fast active learning for machine learning potentials in atomistic and ab initio molecular dynamics simulations. *npj Comput. Mater.*, https://doi.org/10.1038/s41524-025-01897-8
