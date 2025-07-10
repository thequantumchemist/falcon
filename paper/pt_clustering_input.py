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

# During investigation of the effect of clustering into subsets of data, all parameters except for the modelsize and count (max_clusters), were kept constant over all calculations.

modelsize = 500       # Average model sizes of 100, 200 and 500 were tested in thee publication.
max_clusters = 60     # Set to ensure that  modelsize * max_clusters == 30000, matching the total number of MD steps (maximum training structures).


atoms = load_structure('Pt55')   # Load Pt55 cluster as ASE atoms object.
exact_calc = EMT()    # ASE EMT potential is used as calculator for exact calculations during OTF training.
T = 600               # Temperature in K.
accuracy_e = 0.050    # Small accuracy threshold is chosen to have frequent retraining for investigation of clustering.


# Geometry optimization with EMT potential to generate intial training structures

atoms.calc = exact_calc
qn = QuasiNewton(atoms, trajectory='opt.traj')
qn.run(0.001, 10)
training_data = read('opt.traj@0:')



# Setup of the FALCON-OTF-Calculator

atoms.calc = FALCON(model = GPR(atoms),            # All simulations of Pt clusters in the simulation used the default AGOX GPR model.
                    calc = exact_calc,
                    training_data = training_data,
                    accuracy_e = accuracy_e,
                    modelsize = modelsize,
                    max_clusters = max_clusters)

# Setup of the MD Simulation

dyn = Langevin(atoms, 1 * units.fs, temperature_K=T, friction=0.002)


traj = Trajectory(f'MD.traj', 'w', atoms)
dyn.attach(traj.write)                      # Write an ASE Trajectory for every frame.


# Now run the OTF-MD Simulation!
dyn.run(1000000)    # All simulations were performed for one million MD steps (1 ns total simulation time).
