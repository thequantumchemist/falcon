# FALCON Tutorials


This directory contains example scripts demonstrating how to use the **FALCON**
on-the-fly (OTF) calculator for molecular dynamics simulations.


Installation and general setup are described in the main project `README.md`.
This document focuses exclusively on the functionality and execution of the
tutorials contained in this directory.

The directory contains three tutorial scripts, designed to gradually introduce the main concepts of **FALCON**:
1. Basic OTF molecular dynamics with a default ML model.
2. Advanced OTF training with a customized Sparse Gaussian Process model.
3. Postprocessing and analysis of simulation results.

---


## Tutorial Overview

- `simple_tutorial.py`  
  Minimal on-the-fly MD example using the default Gaussian Process model.

- `advanced_tutorial.py`  
  Advanced on-the-fly MD with a custom SparseGPR model and aluminum melting.

- `advanced_tutorial_analysis.py`
  Postprocessing and visualization of the advanced tutorial results.

---

## Tutorial 1: `simple_tutorial.py`  
### Minimal On-the-Fly Molecular Dynamics

This tutorial demonstrates:
- Loading a predefined Pt₅₅ cluster
- Geometry optimization using an exact EMT calculator
- Generation of initial training data
- On-the-fly training of a default GPR model
- Molecular dynamics with a Langevin thermostat

The ML model is trained automatically whenever the predicted energy
uncertainty exceeds a defined threshold.

**Run with:**
```bash
python simple_tutorial.py
```

Output files:
 - ``MD.traj`` - molecular dynamics trajectory
- ``opt.traj`` - geometry optimization trajectory (used as initial training data)

---

## Tutorial 2: `advanced_tutorial.py`
### Advanced On-the-Fly MD and Melting Simulation

This tutorial demonstrates a more realistic and configurable FALCON-MD workflow using a periodic bulk system.
It includes:

- Construction of a periodic FCC aluminum supercell
- Geometry optimization to generate initial training data
- Manual setup of a custom AGOX SparseGPR model
- Energy and force uncertainty thresholds
- Advanced training control (clustering, retraining schedules)
- On-the-fly molecular dynamics during a solid → liquid phase transition

Run with **standard output redirection**:
```bash
python advanced_tutorial.py >> falcon_output.out
```

Redirecting the standard output stream (``stdout``) stores the FALCON log, which contains information about trainings 
and uncertainty decisions. This file is required for the postprocessing.

Output files:
- ``MD.traj`` - molecular dynamics trajectory
- ``OTF_Training_Data.traj`` - accumulated training data
- ``falcon_output.out`` - FALCON log output

---

## Tutorial 3: `advanced_tutorial_analysis.py`
### Postprocessing and Analysis

This postprocessing script analyzes the results of the previous advanced tutorial calculation.
It includes:

- Plotting of the system energy versus the simulation time with highlighted on-the-fly training steps.
- Computation of the radial distribution functions (RDFs) for solid and liquid phases of aluminum

**Required input:**
- ``MD.traj`` - molecular dynamics trajectory
- ``falcon_output.out`` - FALCON log output

**Run with:**
```bash
python advanced_tutorial_analysis.py
```
---

## Recommended Workflow

**1.** Run ``simple_tutorial.py`` to understand the basic FALCON workflow.
**2.** Run ``advanced_tutorial.py > output.out`` for a more complex OTF-MD simulation.
**3.** Analyze the results using ``advanced_tutorial_analysis.py``.

---

# Detailed Tutorial Descriptions

This section provides a **step-by-step explanation of each tutorial script**,
describing the workflow, main components, and how the individual parts interact
during an on-the-fly (OTF) molecular dynamics simulation with **FALCON**.

---

## Detailed Description: `simple_tutorial.py`

### Goal of the Script

The goal of `simple_tutorial.py` is to demonstrate the **core idea of FALCON-MD**
in the simplest possible setting:

A molecular dynamics simulation where a machine-learning model is trained on-the-fly 
and automatically decides when an exact calculation is required.

This script is optimized for clarity and speed rather than realism.

---

### Step 1: Load the atomic structure and define the simulation parameters

```python
atoms = load_structure('Pt55')
```

A predefined Pt₅₅ cluster is loaded using the load_structure() function as the starting model of the simulations.
Additionally, ASE's EMT potential is defined as the calculator for exact calculation during the OTF training and the 
temperature of the simulation and accuracy threshold for retraining is defined.

```python
exact_calc = EMT()
T = 600
accuracy_e = 0.10
```

### Step 2: Generating the initial training data

Before starting MD, the structure is relaxed using the EMT potential and the trajectory of the optimization is used as initial training
data for the ML model.

```python
atoms.calc = exact_calc
qn = QuasiNewton(atoms, trajectory='opt.traj')
qn.run(0.00001, 10)
training_data = read('opt.traj@0:')
```

### Step 3: Setup of the FALCON Calculator 
The core of the script is the setup of the FALCON on-the-fly (OTF) calculator, which in its simplest form requires only 
the four arguments shown below. For the machine learning model, the default Gaussian Progress Regression (GPR) model, 
implemented in [AGOX](https://agox.gitlab.io/agox/index.html) is used.

```python
atoms.calc = FALCON(model = GPR(atoms),            # The default AGOX GPR model is used for this simulation.
                    calc = exact_calc,
                    training_data = training_data,
                    accuracy_e = accuracy_e)
```

### Step 4: Setup of the MD simulation

The MD simulation runs using a Langevin thermostat and the frames are saved to an ASE trajectory file for postprocessing and visulaization.
```python
dyn = Langevin(atoms, 1 * units.fs, temperature_K=T, friction=0.002)
traj = Trajectory(f'MD.traj', 'w', atoms)
dyn.attach(traj.write) 
```

### Step 5: Run the Simulation

Now the MD simulation can be started. However for real runs you should increase thee number of MD steps.
```python
dyn.run(100)    # Number of steps should be increased for real simulations.
```

---

## Detailed Description: `advanced_tutorial.py`

### Goal of the Script

The goal of `advanced_tutorial.py` is to demonstrate a **realistic and highly
configurable on-the-fly molecular dynamics (OTF-MD) workflow** using FALCON-MD,
including:

- Custom machine-learning models
- Force-based uncertainty control
- Long simulations with retraining management
- Phase transitions during molecular dynamics

The example simulates **melting of bulk aluminum**.

---

### Step 1: Build a periodic bulk structure

```python
atoms = FaceCenteredCubic(
    directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    symbol="Al",
    size=(size, size, size),
    pbc=True
)

An FCC aluminum supercell is constructed using ASE’s lattice tools. Periodic boundary conditions are enabled to represent bulk material.

### Step 2: Define exact calculator and accuracy thresholds

```python
exact_calc = EMT()
accuracy_e = 0.10
accuracy_f = 0.50
```
Two independent uncertainty criteria are used:

- ``accuracy_e``: energy uncertainty threshold (eV)
- ``accuracy_f``: maximum force component uncertainty threshold (eV/Å)

If either threshold is exceeded, FALCON triggers an exact calculation and retraining step.

### Step 3: Generate initial training data

As in the simple tutorial, an initial geometry optimization is performed using the exact calculator. 
The resulting structures are physically meaningful and serve as stable initial training data for the ML model.

```python
atoms.calc = exact_calc
qn = QuasiNewton(atoms, trajectory='opt.traj')
qn.run(0.00001, 10)
training_data = read('opt.traj@0:')
```

### Step 4: Manual construction of a SparseGPR model

```python
descriptor = Fingerprint.from_atoms(atoms)      # Global Fingerprint is setup as the descriptor.
kernel = C(1, (1, 100)) * RBF(20, (10, 30))     # Definition of the Kernel used for the SparseGPR model.

model = SparseGPR(kernel=kernel,
                  descriptor=descriptor,
                  sparsifier=CUR(1000),         # Sparsification using CUR algorithm with 1000 sparse points
                  noise_E=0.01,                 # Noise of the energy in eV/atom.
                  noise_F=0.05,                 # Noise of the forces in eV/Å/atom.
                  force_data_filter="none",     # This setting trains the model on energies only. To include forces, set force_data_filter="all". 
                  train_uncertainty=True,       # If True, the ML Model is trained on energy + energy uncertainty.
                  use_ray=True)                 # Ray is used for parallelization.

```

Instead of using the default GPR model, a SparseGPR model from AGOX is manually configured.
Key aspects of this setup include:

- Global fingerprint descriptor
- RBF kernel with tunable hyperparameters
- CUR sparsification to limit model size
- Optional force training
- Parallelization using Ray

This configuration allows scaling to longer simulations and larger datasets.


### Step 5: Advanced FALCON cofiguration
This part of the script shows all possible keywords that can be adjusted for better control of the FALCON OTF training.

```python 
atoms.calc = FALCON(model = model,
                    calc = exact_calc,
                    training_data = training_data,
                    accuracy_e = accuracy_e,
                    accuracy_f = accuracy_f,
                    modelsize = 250,                # Defines the average model size used ifor clustering into subsets of data.
                    max_clusters = 20,              # Defines the maximum number of different ML models that will be used (20 is more than enough here , consuidering the chosen modelsize.)
                    train_start = 10,               # The ML model will be trained in the first 10 MD steps, regardlesss of the uncertainty.
                    train_every = 100,              # The ML model will be trained in every 100th steps. (Default to 1e9.)
                    train_log = True,               # The ML model is retrained at exponentially increasing intervals (2**x steps). In most cases this is the better option than defining train_every.
                    write_training_data = True,     # Training data will be written to the td_filename. This is important for restarting the OTF training.
                    td_filename =  'OTF_Training_Data.traj')

```

### Step 6: Setup and Running the MD simulation

The simulation consists of two stages:

- Equilibration at 300 K (solid aluminum)
- Gradual heating to 3000 K followed by liquid-phase molecular dynamics

Throughout both stages, FALCON continuously monitors uncertainty and retrains the ML model as needed.

```python
MaxwellBoltzmannDistribution(atoms,temperature_K=300)    # Initial velocities of atoms is sampled from a velocity distribution at 300 K. 
Stationary(atoms)
dyn1 = Langevin(atoms, 0.5 * units.fs, friction=0.002, temperature_K=300)
dyn1.attach(traj.write)

dyn2 = Langevin(atoms, 0.5 * units.fs, friction=0.002, temperature_K=3000, tbegin=300, heatsteps=2000)
dyn2.attach(traj.write, interval=1)

dyn1.run(6000)      # 3 ps of equillibration at 300 K. 
dyn2.run(20000)     # 1 ps (2000 MD steps) of heating from 300 K to 3000 K, followed by 9 ps at 3000 K.
```
---

## Detailed Description: `advanced_tutorial_analysis.py`

### Goal of the Script
This analysis script demonstrates how to extract physical insight from an OTF-MD simulation performed with FALCON-MD.
It relies on:
- The MD trajectory (``MD.traj``)
- The FALCON log output (``falcon_output.out``)

### Step 1: Energy and training event visualization
The first function:
- Plots total energy versus simulation time
- Marks MD steps where FALCON retraining occurred
- Allows correlation of training events with physical changes


```python
plot_simulation("MD.traj", "falcon_output.out")
```

### Step 2: Calculation and visualization of the RDF
The radial distribution function (RDF) is averaged over two time windows:
- One corresponding to the solid phase
- One corresponding to the liquid phase

This provides a clear structural comparison between ordered and disordered states.

```python
plot_rdf(filename="MD.traj",
         index_solid="3000:5000:200",
         index_liquid="20000:22000:200")
```



