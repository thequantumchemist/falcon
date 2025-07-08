from ase.calculators.emt import EMT
from ase import units
from ase.io.trajectory import Trajectory
from ase.optimize import QuasiNewton
from ase.io import read, write

from falcon.otf_calculator import OTFCalculator
from falcon.models.agox_models import GPR
from falcon.structures import load_structure
from falcon.utils.langevin import Langevin

##########################################################################

# General Setup

atoms = load_structure('Pt55')   # Setup ASE atoms object (Structure is loaded from FALCON's tutorial structures.)

exact_calc = EMT()   # Calculator for exact calculations during OTF training.
T = 600              # Temperature in K.
accuracy_e = 0.10    # Accuracy Threshold (Epsilon) in eV.



# Geometry optimization with EMT potential to generate intial training structures

atoms.calc = exact_calc
qn = QuasiNewton(atoms, trajectory='opt.traj')
qn.run(0.001, 10)
training_data = read('opt.traj@0:')



# Setup of the FALCON-OTF-Calculator

atoms.calc = OTFCalculator(model = GPR(atoms),            # The default AGOX GPR model is used for this simulation.
                           calc = exact_calc,
                           training_data = training_data,
                           accuracy_e = accuracy_e)

# Setup of the MD Simulation

dyn = Langevin(atoms, 1 * units.fs, temperature_K=T, friction=0.002)


traj = Trajectory(f'MD.traj', 'w', atoms)
dyn.attach(traj.write)                      # An ASE Trajectory will be written for every frame (can be changed using keyword 'interval=')


# Now run the OTF-MD Simulation!
dyn.run(100)    # Number of steps should be increased for real simulations.
