from ase.calculators.emt import EMT
from ase import units
from ase.io.trajectory import Trajectory
from ase.optimize import QuasiNewton
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.lattice.cubic import FaceCenteredCubic

from falcon-md.otf_calculator import FALCON
from falcon-md.utils.langevin import Langevin

##########################################################################

# General Setup

# Setup ASE atoms object (fcc-Al supercell is built using ASE's building tools.)
size = 2
atoms = FaceCenteredCubic(directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                          symbol="Al",
                          size=(size, size, size),
                          pbc=True)


exact_calc = EMT()   # Calculator for exact calculations during OTF training.
accuracy_e = 0.10    # Accuracy Threshold (Epsilon) in eV.
accuracy_f = 0.50    # Accuracy Threshold of maximum Force component in eV/Å.


# Geometry optimization with EMT potential to generate intial training structures

atoms.calc = exact_calc
qn = QuasiNewton(atoms, trajectory='opt.traj')
qn.run(0.00001, 10)
training_data = read('opt.traj@0:')


# Instead of importing a default GPR Model we manually setup AGOX SparseGPR model.

from agox.models.descriptors import Fingerprint
from agox.models.GPR import SparseGPR
from agox.models.GPR.kernels import RBF, Noise
from agox.models.GPR.kernels import Constant as C
from agox.models.GPR.priors import Repulsive
from agox.environments import Environment
from agox.utils.sparsifiers import CUR


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


# Setup of the FALCON-OTF-Calculator

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



# Setup of the MD Simulation

traj = Trajectory(f'MD.traj', 'w', atoms)    # Setup ASE trajectory to save the MD frames.

MaxwellBoltzmannDistribution(atoms,temperature_K=300)    # Initial velocities of atoms is sampled from a velocity distribution at 300 K. 
Stationary(atoms)

# Initial Simulation at 300 K (in solid phase).
dyn1 = Langevin(atoms, 0.5 * units.fs, friction=0.002, temperature_K=300)
dyn1.attach(traj.write)

# Melting of the structure (Temperature ios increased to 3000 K.
dyn2 = Langevin(atoms, 0.5 * units.fs, friction=0.002, temperature_K=3000, tbegin=300, heatsteps=2000)
dyn2.attach(traj.write, interval=1)


# Now run the OTF-MD Simulation!
dyn1.run(6000)      # 3 ps of equillibration at 300 K. 
dyn2.run(20000)     # 1 ps (2000 MD steps) of heating from 300 K to 3000 K, followed by 9 ps at 3000 K.



