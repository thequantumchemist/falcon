from ase.calculators.emt import EMT
from ase import units
from ase.io.trajectory import Trajectory
from ase.optimize import QuasiNewton
from ase.io import read, write

from falcon_md.otf_calculator import FALCON
from falcon_md.models.agox_models import GPR
from falcon_md.structures import load_structure
from falcon_md.utils.langevin import Langevin

##########################################################################

# General Setup

atoms = load_structure('Pt55')   # Load Pt55 cluster as ASE atoms object. Pt13, Pt147, and Pt561 are also available.

exact_calc = EMT()    # ASE EMT potential is used as calculator for exact calculations during OTF training.
T = 600               # Temperature in K. (Temperatures were varied for different calculations in the publication.)
accuracy_e = 0.125    # Accuracy threshold (epsilon) in eV. (Thresholds were varied for different calculations in the publication.)



# Geometry optimization with EMT potential to generate intial training structures

atoms.calc = exact_calc
qn = QuasiNewton(atoms, trajectory='opt.traj')
qn.run(0.001, 10)
training_data = read('opt.traj@0:')



# Setup of the FALCON-OTF-Calculator

atoms.calc = FALCON(model = GPR(atoms),            # All simulations of Pt clusters in the simulation used the default AGOX GPR model.
                    calc = exact_calc,
                    training_data = training_data,
                    accuracy_e = accuracy_e)

# Setup of the MD Simulation

dyn = Langevin(atoms, 1 * units.fs, temperature_K=T, friction=0.002)


traj = Trajectory(f'MD.traj', 'w', atoms)
dyn.attach(traj.write)                      # An ASE Trajectory will be written for every frame (can be changed using keyword 'interval=')


# Now run the OTF-MD Simulation!
dyn.run(1000000)    # All simulations were performed for one million MD steps (1 ns total simulation time).
