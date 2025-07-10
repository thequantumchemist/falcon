from ase.calculators.vasp import Vasp
from ase import units
from ase.io.trajectory import Trajectory
from ase.optimize import QuasiNewton
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.lattice.cubic import FaceCenteredCubic

from falcon_md.otf_calculator import FALCON
from falcon_md.utils.langevin import Langevin

##########################################################################

# General Setup

accuracy_e = 0.05    # Accuracy thresholds of 0.05 eV and 0.10 eV were used for the simulations of Al in the publication.

# Setup ASE atoms object (fcc-Al supercell is built using ASE's building tools.)
size = 2
atoms = FaceCenteredCubic(directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                          symbol="Al",
                          size=(size, size, size),
                          pbc=True)


# Setup VASP Calculator for DFT Calculations
exact_calc = Vasp(label='vaspcalc/vaspinp',
                  xc='PBE', 
                  nsw=0,
                  kpts=[2,2,2],
                  lreal=False, 
                  ncore=8,kpar=4,
                  nelm=500,
                  isym=0,
                  encut=415,
                  potim=0.5,
                  ediff=0.0001)



# Geometry optimization to generate intial training structures
atoms.calc = exact_calc
qn = QuasiNewton(atoms, trajectory='opt.traj')
qn.run(0.00001, 2)
training_data = read('opt.traj@0:')


# Setup of the ML model (All three models were used in for simulations mentioned in the publication).

# AGOX GPR model trained only on energies.
from falcon_md.models.agox_models import GPR
model = GPR(atoms)


# AGOX Sparse GPR model trained on energies and forces.
#from falcon_md.models.agox_models import SparseGPR
#model = SparseGPR(atoms)


# AGOX GPR model trained on energies and forces.
#from falcon_md.models.agox_models import GPRForces
#model = GPRForces(atoms)



# Setup of the FALCON-OTF-Calculator

atoms.calc = FALCON(model = model,
                    calc = exact_calc,
                    training_data = training_data,
                    accuracy_e = accuracy_e)



# Setup of the MD Simulation

traj = Trajectory(f'MD.traj', 'w', atoms)    # Setup ASE trajectory to save the MD frames.

MaxwellBoltzmannDistribution(atoms,temperature_K=300)    # Initial velocities of atoms is sampled from a velocity distribution at 300 K. 
Stationary(atoms)

# Initialization of the Simulation at 300 K (in solid phase).
dyn1 = Langevin(atoms, 0.5 * units.fs, friction=0.002, temperature_K=300)
dyn1.attach(traj.write)

# Melting of the structure (Temperature is increased to 3000 K.)
dyn2 = Langevin(atoms, 0.5 * units.fs, friction=0.002, temperature_K=3000, tbegin=300, heatsteps=2000)
dyn2.attach(traj.write, interval=1)


# Now run the OTF-MD Simulation!
dyn1.run(10000)      # 5 ps of equillibration at 300 K. 
dyn2.run(500000)     # 1 ps (2000 MD steps) of heating from 300 K to 3000 K, followed by 249 ps at 3000 K.


