from ase.calculators.vasp import Vasp
from ase import units
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE as BFGS
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from falcon_md.otf_calculator import FALCON
from falcon_md.models.agox_models import GPR
from falcon_md.structures import load_structure
from falcon_md.utils.langevin import Langevin

##########################################################################

# General Setup

atoms = load_structure('CNT_H2O')    # Load structure of carbon nanotuube with water molecule  as ASE atoms object.
T0 = 10                              # Starting temperature in K.
T1 = 400                             # End temperature in K.
accuracy_e = 0.01                    # Accuracy threshold (epsilon) in eV.

# Setup of the VASP Calculator for DFT calculations
exact_calc = Vasp(label = 'vaspcalc/vaspinp',
                  xc = 'PBE', 
                  nsw = 0,
                  kpts = [1,1,1],
                  lreal = False, 
                  ncore = 8,
                  kpar = 4,
                  nelm = 500,
                  isym = 0,
                  encut = 415,
                  potim = 0.5,
                  ediff = 0.0001)


# Geometry optimization with DFT  to generate intial training structures

atoms.calc = exact_calc
opt = BFGS(atoms, maxstep=0.1, trajectory='opt.traj')
opt.run(fmax=0.05, steps=3)
training_data = read('opt.traj@0:')



# Setup of the FALCON-OTF-Calculator

atoms.calc = FALCON(model = GPR(atoms),            # The default AGOX GPR model is used for th simulation.
                    calc = exact_calc,
                    training_data = training_data,
                    accuracy_e = accuracy_e)

# Setup of the MD Simulation

MaxwellBoltzmannDistribution(atoms, temperature_K=T0)    # Initialization at 10 K.
Stationary(atoms)


dyn = Langevin(atoms, 0.5 * units.fs, friction=0.002, temperature_K=T1, tbegin=T0, heatsteps=1000)

traj = Trajectory(f'MD.traj', 'w', atoms)
dyn.attach(traj.write)                      # Write an  ASE Trajectory for every frame.


dyn.run(50000)    # Simulate for 50000 MD steps (25 ps total simulation time).
