# FALCON
This work presents the __FALCON__ (**F**ast __A__ctive __L__earning for __C__omputational ab initio m__O__lecular dy__N__amics) calculator where the ML model is trained on-the-fly (OTF) and uses its own uncertainty estimation to decide whether an exact calculation is required. 

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
## Simple tutorial
A basic tutorial script for molecular dynamics simulation with FALCON can be found at `~/falcon_md/tutorial/simple_tutorial.py`.
This example simulates a Pt₅₅ cluster using a simple Effective Medium Theory (EMT) potential, allowing a fast simulation to without requiring extensive computational resources.

The interesting section of the script starts with loading the structure of the Pt₅₅ cluster using the load_structure() function, which returns an ASE atoms object as the basis of the simulation.
Additionally,  ASE's EMT potential is defined as the calculator for exact calculation during the OTF training and the temperature of the simulation and accuracsy threshld for retraining is defined.

```bash
# General Setup

atoms = load_structure('Pt55')   # Setup ASE atoms object (Structure is loaded from FALCON's tutorial structures.)

exact_calc = EMT()   # Calculator for exact calculations during OTF training.
T = 600              # Temperature in K.
accuracy_e = 0.10    # Accuracy Threshold (Epsilon) in eV.
```


Before starting MD, the structure is relaxed using the EMT potential and the trajectory of the optimization is used as initial training data for the ML model.

```bash
# Geometry optimization with EMT potential to generate intial training structures

atoms.calc = exact_calc
qn = QuasiNewton(atoms, trajectory='opt.traj')
qn.run(0.001, 10)
training_data = read('opt.traj@0:')
```

The core of the script is the setup of the FALCON on-the-fly (OTF) calculator, which in its simplest form requires only the four arguments shown below.
For the machine learning model, the default Gaussian Progress Regression (GPR) model, implemented in [AGOX](https://agox.gitlab.io/agox/index.html) is used.

```bash
# Setup of the FALCON-OTF-Calculator

atoms.calc = FALCON(model = GPR(atoms),            # The default AGOX GPR model is used for this simulation.
                    calc = exact_calc,
                    training_data = training_data,
                    accuracy_e = accuracy_e)
```

The MD simulation runs using a Langevin thermostat and the frames are saved to an ASE trajectory file for postprocessing and visulaization.


```bash
# Setup of the MD Simulation

dyn = Langevin(atoms, 1 * units.fs, temperature_K=T, friction=0.002)


traj = Trajectory(f'MD.traj', 'w', atoms)
dyn.attach(traj.write)                      # An ASE Trajectory will be written for every frame (can be changed using keyword 'interval=')
```

Now the MD simulation can be started. However for real runs you should increase thee number of MD steps.
```bash
# Now run the OTF-MD Simulation!
dyn.run(100)    # Number of steps should be increased for real simulations.
```

## Advanced tutorial
An advanced tutorial script is available at `~/falcon_md/tutorial/advanced_tutorial.py`.

The advanved tutorial demonstrates a molecular dynamics simulation of aluminum melting, again using the EMT potential. It showcases the various parameters that can be adjusted for better control of the FALCON OTF training.

After running the advanced tutorial you can analyse the results by visualising the simulation progression and Radial Distribution Functions, using the `~/falcon_md/tutorial/advanced_tutorial_analysis.py` script. 


------------

# Citation
When using FALCON, please cite the following papers:
