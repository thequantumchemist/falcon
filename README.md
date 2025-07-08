# Falcon
This work presents the FALCON (Fast Active Learning for Computational ab initio mOlecular dyNamics) calculator where the ML model is trained on-the-fly (OTF) and uses its own uncertainty estimation to decide whether an exact calculation is required. 

![Alt text](FALCON_MD.png?raw=true "FALCON")


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
FALCON can be installed by running ``pip install falcon-md``. When using the ``git`` version, add ``~/falcon-md`` to your $PYTHONPATH environment variable. 

------------

# Tutorial
A tutorial is available here: https://thequantumchemist.github.io/falcon/

------------

# Citation
When using FALCON, please cite the following papers:
