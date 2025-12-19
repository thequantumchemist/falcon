import numpy as np
from dscribe.descriptors import SOAP, LMBTR, MBTR, ACSF, CoulombMatrix, ValleOganov
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

"""
=============================================================================
 EXAMPLES for descriptor_params:

 SOAP:
   descriptor_params = {
       "species": ["H", "O"],
       "periodic": False,
       "r_cut": 5,
       "n_max": 8,
       "l_max": 6,
       "sigma": 1.0,
       "sparse": False,
   }

 MBTR or LMBTR:
   descriptor_params = {
       "species": ["H", "O"],
       "periodic": False,
       "geometry": {"function": "distance"},
       "grid": {"min": 0, "max": 5, "n": 200, "sigma": 0.1},
       "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-4},
       "normalization": "l2",
   }

 ACSF:
   descriptor_params = {
       "species": ["H", "O"],
       "periodic": False,
       "r_cut": 3.5,
       "g2_params": [[1, 1], [1, 2], [1, 3]],
       "g4_params": [[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]]
   }

 CoulombMatrix:
   descriptor_params = {
       "n_atoms_max": 3,
       "flatten": True,
       "sparse": False
   }

 ValleOganov:
   descriptor_params = {
       "atom_types": ["H", "O"],
       "cutoff": 6.0,
       "flatten": True,
       "sparse": False
   }

 =============================================================================
"""

class DescriptorWrapper:
    """
    Wrapper for DScribe descriptors.
    Supported descriptors:
      - "soap"
      - "mbtr"
      - "lmbtr"
      - "acsf"
      - "coulombmatrix"
      - "valleoganov"

    Example:
      wrapper = DescriptorWrapper(atoms, descriptor="valleoganov", descriptor_params={})
      X = wrapper.get_features([atoms])[0]   # (1, n_features)
      x_single = wrapper.create_features(atoms) # [np.ndarray]
    """
    
    def __init__(self, atoms_example, descriptor='soap', descriptor_params=None):
        unique_species = list(set(atoms_example.get_chemical_symbols()))
        periodic = atoms_example.get_pbc().any()
        if descriptor_params is None:
            descriptor_params = {}
        if descriptor == 'mbtr':
            defaults = dict(
                species=unique_species,
                geometry={"function": "distance"},
                grid={"min": 0, "max": 5, "n": 200, "sigma": 0.1},
                weighting={"function": "exp", "scale": 0.5, "threshold": 1e-4},
                periodic=periodic,
                normalization="l2"
            )
            defaults.update(descriptor_params)
            self.desc = MBTR(**defaults)
        elif descriptor == 'lmbtr':
            defaults = dict(
                species=unique_species,
                geometry={"function": "distance"},
                grid={"min": 0, "max": 5, "n": 200, "sigma": 0.1},
                weighting={"function": "exp", "scale": 0.5, "threshold": 1e-4},
                periodic=periodic,
                normalization="l2"
            )
            defaults.update(descriptor_params)
            self.desc = LMBTR(**defaults)
        elif descriptor == 'acsf':
            defaults = dict(
                species=unique_species,
                periodic=periodic,
                r_cut=5.5,
                g2_params=[[1, 1], [1, 2], [1, 3]],
                g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]]
            )
            defaults.update(descriptor_params)
            self.desc = ACSF(**defaults)
        elif descriptor == 'coulombmatrix':
            n_atoms_max = descriptor_params.get('n_atoms_max', len(atoms_example))
            permutation = descriptor_params.get('permutation', 'sorted_l2')
            self._cm_flatten = descriptor_params.get('flatten', True)
            self.desc = CoulombMatrix(n_atoms_max=n_atoms_max, permutation=permutation)

        elif descriptor == 'valleoganov':
            if 'sigma' in descriptor_params:
                sigma = descriptor_params['sigma']
            else:
                sigma=10**(-0.5)
            defaults = dict(
                species=unique_species,
                sigma=sigma,
                function='distance',
                r_cut=3.5,
                n=100
            )
            defaults.update(descriptor_params)
            self.desc = ValleOganov(**defaults)
        else:
            # SOAP as default
            defaults = dict(
                species=unique_species,
                periodic=periodic,
                r_cut=5,
                n_max=8,
                l_max=6,
                sigma=1.0,
                sparse=False
            )
            defaults.update(descriptor_params)
            self.desc = SOAP(**defaults)

    def get_features(self, structures):
        """Gibt [2D-Array] zurück, eine Zeile pro Struktur."""
        return [np.array([self.desc.create(atoms).reshape(-1) for atoms in structures])]

    def create_features(self, atoms):
        """Gibt Einzelvector als Liste für sklearn .predict()."""
        return [self.desc.create(atoms).reshape(-1)]



class ASE_DistanceMatrix_Wrapper:
    """
    Feature wrapper for using the symmetric minimum-image convention (MIC) distance matrix
    as a sorted, flattened descriptor vector.
    
    Usage:
        - .get_features([Atoms, ...])  -> [array of shape (n_structures, n_features)]
        - .create_features(Atoms)      -> [array of shape (n_features,)]
    """
    def __init__(self, atoms_example, descriptor_params=None):
        # Use mic=True if the system is periodic, else False.
        self.periodic = atoms_example.get_pbc().any()

    def get_single_flat_distance(self, atoms):
        """
        Gets the sorted, unique pairwise distances for a structure (excluding diagonal).
        Output: 1D array of length N*(N-1)//2, sorted.
        """
        D = atoms.get_all_distances(mic=self.periodic)
        idx_triu = np.triu_indices_from(D, k=1)
        distances = D[idx_triu]
#        return np.sort(distances)  # Ensures order-invariance
        return distances  # Ensures order-invariance

    def get_features(self, structures):
        """
        Returns a list containing an array (n_structures, n_features)
        (Compatibel for FALCON's get_features interface).
        """
        features = [self.get_single_flat_distance(atoms) for atoms in structures]
        return [np.stack(features)]

    def create_features(self, atoms):
        """
        Returns single structure descriptor vector as [array] for ML model prediction.
        """
        return [self.get_single_flat_distance(atoms)]






class SKLEARN_GPR:
    """
    Gaussian Process Regression compatible with DScribe descriptors.
    - Usage: model = SKLEARN_GPR(atoms, descriptor="soap", descriptor_params=...)
    - Default: train_forces=False
    """

    def __init__(self, atoms_example, descriptor='soap', descriptor_params=None,
                 train_forces=False, fd_step=1e-2):
        if descriptor=='distancematrix':
            self.descriptor = ASE_DistanceMatrix_Wrapper(atoms_example)
        else:
            self.descriptor = DescriptorWrapper(atoms_example, descriptor, descriptor_params)
        kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-6)
        self.gpr_e = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)
        self.train_forces = train_forces
        self.fd_step = fd_step
        if self.train_forces:
            self.gpr_f = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)
        self._cache_atoms_id = None
        self._cache_energy = None
        self._cache_uncertainty = None
        self.trained = False

    def train(self, structures):
        X = self.descriptor.get_features(structures)[0]
        energies = np.array([atoms.get_potential_energy() for atoms in structures])
        self.gpr_e.fit(X, energies)
        if self.train_forces:
            forces = np.array([atoms.get_forces().flatten() for atoms in structures])
            self.gpr_f.fit(X, forces)
        self.trained = True

    def _predict_energy_uncertainty(self, atoms):
        atoms_id = (id(atoms), atoms.get_positions().tobytes())
        if atoms_id == self._cache_atoms_id:
            return self._cache_energy, self._cache_uncertainty
        feat = self.descriptor.create_features(atoms)
        y_mean, y_std = self.gpr_e.predict(feat, return_std=True)
        self._cache_atoms_id = atoms_id
        self._cache_energy = y_mean[0]
        self._cache_uncertainty = y_std[0]
        return self._cache_energy, self._cache_uncertainty

    def predict_energy(self, atoms):
        """Mache eine Energievorhersage für ein Atoms-Objekt."""
        e, _ = self._predict_energy_uncertainty(atoms)
        return e

    def predict_uncertainty(self, atoms):
        """Unsicherheit der Energie (predict_uncertainty)"""
        _, u = self._predict_energy_uncertainty(atoms)
        return u

    def _single_force_unc(self, atoms, i, j, positions, delta):
        atoms_p = atoms.copy()
        pos_p = positions.copy()
        pos_p[i, j] += delta
        atoms_p.set_positions(pos_p)
        e_p, u_p = self._predict_energy_uncertainty(atoms_p)
        atoms_m = atoms.copy()
        pos_m = positions.copy()
        pos_m[i, j] -= delta
        atoms_m.set_positions(pos_m)
        e_m, u_m = self._predict_energy_uncertainty(atoms_m)
        force = -(e_p - e_m) / (2 * delta)
        force_unc = abs(-(u_p - u_m) / (2 * delta))
        return (i, j, force, force_unc)

    def numerical_forces_and_uncertainty(self, atoms):
        positions = atoms.get_positions()
        n_atoms = len(atoms)
        delta = self.fd_step
        displaced_atoms = []
        for i in range(n_atoms):
            for j in range(3):
                for sign in [-1,1]:
                    atom_disp = atoms.copy()
                    pos_disp = positions.copy()
                    pos_disp[i, j] += sign * delta
                    atom_disp.set_positions(pos_disp)
                    displaced_atoms.append(atom_disp)
        X_batch = np.array([self.descriptor.create_features(a)[0] for a in displaced_atoms])
        y_batch, std_batch = self.gpr_e.predict(X_batch, return_std=True)
        forces = np.zeros_like(positions)
        force_unc = np.zeros_like(positions)
        idx = 0
        for i in range(n_atoms):
            for j in range(3):
                e_m, e_p = y_batch[idx], y_batch[idx+1]
                u_m, u_p = std_batch[idx], std_batch[idx+1]
                f = -(e_p - e_m) / (2 * delta)
                fu = abs(-(u_p - u_m) / (2 * delta))
                forces[i, j] = f
                force_unc[i, j] = fu
                idx += 2
        self.force_unc=force_unc
        return forces, force_unc

    def predict_forces(self, atoms):
        if self.train_forces:
            feat = self.descriptor.create_features(atoms)
            f_pred, _ = self.gpr_f.predict(feat, return_std=True)
            return f_pred[0].reshape(-1, 3)
        else:
            forces, _ = self.numerical_forces_and_uncertainty(atoms)
            return forces

    def predict_uncertainty_forces(self, atoms):
        if self.train_forces:
            feat = self.descriptor.create_features(atoms)
            _, f_std = self.gpr_f.predict(feat, return_std=True)
            return f_std[0].reshape(-1, 3)
        else:
            force_unc=self.force_unc
            return force_unc



class SKLEARN_KRR:
    """
    Kernel Ridge Regression ensemble
    Default: train_forces=False 
    """
    def __init__(self, atoms_example, descriptor='soap', descriptor_params=None, train_forces=False, fd_step=1e-2):
        if descriptor=='distancematrix':
            self.descriptor = ASE_DistanceMatrix_Wrapper(atoms_example)
        else:
            self.descriptor = DescriptorWrapper(atoms_example, descriptor, descriptor_params)
        self.reg1 = KernelRidge(alpha=1e-5, kernel='rbf')
        self.reg2 = KernelRidge(alpha=1e-5, kernel='poly', degree=4, coef0=1)
        self.train_forces = train_forces
        self.fd_step = fd_step
        if self.train_forces:
            self.force_reg1 = KernelRidge(alpha=1e-5, kernel='rbf')
            self.force_reg2 = KernelRidge(alpha=1e-5, kernel='poly', degree=4, coef0=1)
        self.trained = False
        self._cache_atoms_id = None
        self._cache_energy = None
        self._cache_uncertainty = None

    def train(self, structures):
        X = self.descriptor.get_features(structures)[0]
        energies = np.array([atoms.get_potential_energy() for atoms in structures])
        self.reg1.fit(X, energies)
        self.reg2.fit(X, energies)
        if self.train_forces:
            forces = np.array([atoms.get_forces().flatten() for atoms in structures])
            self.force_reg1.fit(X, forces)
            self.force_reg2.fit(X, forces)
        self.trained = True

    def _predict_energy_uncertainty(self, atoms):
        atoms_id = (id(atoms), atoms.get_positions().tobytes())
        if atoms_id == self._cache_atoms_id:
            return self._cache_energy, self._cache_uncertainty
        feat = self.descriptor.create_features(atoms)
        e1 = self.reg1.predict(feat)[0]
        e2 = self.reg2.predict(feat)[0]
        mean_e = 0.5 * (e1 + e2)
        unc_e = abs(e1 - e2)
        self._cache_atoms_id = atoms_id
        self._cache_energy = mean_e
        self._cache_uncertainty = unc_e
        return mean_e, unc_e

    def predict_energy(self, atoms):
        mean_e, _ = self._predict_energy_uncertainty(atoms)
        return mean_e

    def predict_uncertainty(self, atoms):
        _, unc_e = self._predict_energy_uncertainty(atoms)
        return unc_e

    def _single_force_unc(self, atoms, i, j, positions, delta):
        atoms_p = atoms.copy()
        pos_p = positions.copy()
        pos_p[i, j] += delta
        atoms_p.set_positions(pos_p)
        e_p, u_p = self._predict_energy_uncertainty(atoms_p)
        atoms_m = atoms.copy()
        pos_m = positions.copy()
        pos_m[i, j] -= delta
        atoms_m.set_positions(pos_m)
        e_m, u_m = self._predict_energy_uncertainty(atoms_m)
        force = -(e_p - e_m) / (2 * delta)
        force_unc = abs(-(u_p - u_m) / (2 * delta))
        return (i, j, force, force_unc)

    def numerical_forces_and_uncertainty(self, atoms):
        """
        Batch-vectorized calculation of the forces and force uncertainties of the KRR ensemble.
        """
        positions = atoms.get_positions()
        n_atoms = len(atoms)
        delta = self.fd_step

        displaced_atoms = []
        for i in range(n_atoms):
            for j in range(3):
                for sign in [-1, 1]:
                    atom_disp = atoms.copy()
                    pos_disp = positions.copy()
                    pos_disp[i, j] += sign * delta
                    atom_disp.set_positions(pos_disp)
                    displaced_atoms.append(atom_disp)

        X_batch = np.array([self.descriptor.create_features(a)[0] for a in displaced_atoms])

        e1_batch = self.reg1.predict(X_batch)
        e2_batch = self.reg2.predict(X_batch)

        forces = np.zeros_like(positions)
        force_unc = np.zeros_like(positions)
        idx = 0
        for i in range(n_atoms):
            for j in range(3):
                e1_m, e1_p = e1_batch[idx], e1_batch[idx+1]
                e2_m, e2_p = e2_batch[idx], e2_batch[idx+1]
                f1 = -(e1_p - e1_m) / (2 * delta)
                f2 = -(e2_p - e2_m) / (2 * delta)
                meanf = 0.5 * (f1 + f2)
                varf = abs(f1 - f2)
                forces[i, j] = meanf
                force_unc[i, j] = varf
                idx += 2
        self.force_unc=force_unc
        return forces, force_unc

    def predict_forces(self, atoms):
        if self.train_forces:
            feat = self.descriptor.create_features(atoms)
            f1 = self.force_reg1.predict(feat)[0]
            f2 = self.force_reg2.predict(feat)[0]
            avg = 0.5 * (f1 + f2)
            return avg.reshape(-1, 3)
        else:
            forces, _ = self.numerical_forces_and_uncertainty(atoms)
            return forces

    def predict_uncertainty_forces(self, atoms):
        if self.train_forces:
            feat = self.descriptor.create_features(atoms)
            f1 = self.force_reg1.predict(feat)[0]
            f2 = self.force_reg2.predict(feat)[0]
            diff = abs(f1 - f2)
            return diff.reshape(-1, 3)
        else:
            force_unc=self.force_unc
            return force_unc





class SKLEARN_SVR:
    """
    SKLEARN Support Vector Regression ensemble (mit model disagreement uncertainty!)
    Default: train_forces=False
    """
    def __init__(self, atoms_example, descriptor='soap', descriptor_params=None, train_forces=False, fd_step=1e-2):
        if descriptor == 'distancematrix':
            self.descriptor = ASE_DistanceMatrix_Wrapper(atoms_example)
        else:
            self.descriptor = DescriptorWrapper(atoms_example, descriptor, descriptor_params)
        self.reg1 = SVR(C=1.0, epsilon=1e-4, kernel='rbf')
        self.reg2 = SVR(C=1.0, epsilon=1e-4, kernel='poly', degree=4, coef0=1)
        self.train_forces = train_forces
        self.fd_step = fd_step
        if self.train_forces:
            self.force_reg1 = SVR(C=1.0, epsilon=1e-4, kernel='rbf')
            self.force_reg2 = SVR(C=1.0, epsilon=1e-4, kernel='poly', degree=4, coef0=1)
        self.trained = False
        self._cache_atoms_id = None
        self._cache_energy = None
        self._cache_uncertainty = None

    def train(self, structures):
        X = self.descriptor.get_features(structures)[0]
        energies = np.array([atoms.get_potential_energy() for atoms in structures])
        self.reg1.fit(X, energies)
        self.reg2.fit(X, energies)
        if self.train_forces:
            forces = np.array([atoms.get_forces().flatten() for atoms in structures])
            self.force_reg1.fit(X, forces)
            self.force_reg2.fit(X, forces)
        self.trained = True

    def _predict_energy_uncertainty(self, atoms):
        atoms_id = (id(atoms), atoms.get_positions().tobytes())
        if atoms_id == self._cache_atoms_id:
            return self._cache_energy, self._cache_uncertainty
        feat = self.descriptor.create_features(atoms)
        e1 = self.reg1.predict(feat)[0]
        e2 = self.reg2.predict(feat)[0]
        mean_e = 0.5 * (e1 + e2)
        unc_e = abs(e1 - e2)
        self._cache_atoms_id = atoms_id
        self._cache_energy = mean_e
        self._cache_uncertainty = unc_e
        return mean_e, unc_e

    def predict_energy(self, atoms):
        mean_e, _ = self._predict_energy_uncertainty(atoms)
        return mean_e

    def predict_uncertainty(self, atoms):
        _, unc_e = self._predict_energy_uncertainty(atoms)
        return unc_e

    def _single_force_unc(self, atoms, i, j, positions, delta):
        atoms_p = atoms.copy()
        pos_p = positions.copy()
        pos_p[i, j] += delta
        atoms_p.set_positions(pos_p)
        e_p, u_p = self._predict_energy_uncertainty(atoms_p)
        atoms_m = atoms.copy()
        pos_m = positions.copy()
        pos_m[i, j] -= delta
        atoms_m.set_positions(pos_m)
        e_m, u_m = self._predict_energy_uncertainty(atoms_m)
        force = -(e_p - e_m) / (2 * delta)
        force_unc = abs(-(u_p - u_m) / (2 * delta))
        return (i, j, force, force_unc)

    def numerical_forces_and_uncertainty(self, atoms):
        positions = atoms.get_positions()
        n_atoms = len(atoms)
        delta = self.fd_step

        displaced_atoms = []
        for i in range(n_atoms):
            for j in range(3):
                for sign in [-1, 1]:
                    atom_disp = atoms.copy()
                    pos_disp = positions.copy()
                    pos_disp[i, j] += sign * delta
                    atom_disp.set_positions(pos_disp)
                    displaced_atoms.append(atom_disp)

        X_batch = np.array([self.descriptor.create_features(a)[0] for a in displaced_atoms])

        e1_batch = self.reg1.predict(X_batch)
        e2_batch = self.reg2.predict(X_batch)

        forces = np.zeros_like(positions)
        force_unc = np.zeros_like(positions)
        idx = 0
        for i in range(n_atoms):
            for j in range(3):
                e1_m, e1_p = e1_batch[idx], e1_batch[idx+1]
                e2_m, e2_p = e2_batch[idx], e2_batch[idx+1]
                f1 = -(e1_p - e1_m) / (2 * delta)
                f2 = -(e2_p - e2_m) / (2 * delta)
                meanf = 0.5 * (f1 + f2)
                varf = abs(f1 - f2)
                forces[i, j] = meanf
                force_unc[i, j] = varf
                idx += 2
        self.force_unc = force_unc
        return forces, force_unc

    def predict_forces(self, atoms):
        if self.train_forces:
            feat = self.descriptor.create_features(atoms)
            f1 = self.force_reg1.predict(feat)[0]
            f2 = self.force_reg2.predict(feat)[0]
            avg = 0.5 * (f1 + f2)
            return avg.reshape(-1, 3)
        else:
            forces, _ = self.numerical_forces_and_uncertainty(atoms)
            return forces

    def predict_uncertainty_forces(self, atoms):
        if self.train_forces:
            feat = self.descriptor.create_features(atoms)
            f1 = self.force_reg1.predict(feat)[0]
            f2 = self.force_reg2.predict(feat)[0]
            diff = abs(f1 - f2)
            return diff.reshape(-1, 3)
        else:
            force_unc = self.force_unc
            return force_unc


from sklearn.neural_network import MLPRegressor
import numpy as np

class SKLEARN_MLP:
    """
    Multi-layer Perceptron ensemble (model disagreement for uncertainty).
    Default: train_forces=False (similar to KRR enesemble).
    """
    def __init__(self, atoms_example, descriptor='soap', descriptor_params=None, train_forces=False, fd_step=1e-2):
        if descriptor=='distancematrix':
            self.descriptor = ASE_DistanceMatrix_Wrapper(atoms_example)
        else:
            self.descriptor = DescriptorWrapper(atoms_example, descriptor, descriptor_params)
        # Unterschiedliche Architekturen und Seeds für "disagreement"
        self.reg1 = MLPRegressor(hidden_layer_sizes=(50,10), activation='relu', alpha=1e-4, random_state=123, max_iter=2000)
        self.reg2 = MLPRegressor(hidden_layer_sizes=(40,4), activation='relu', alpha=1e-5, random_state=456, max_iter=2000)
        self.train_forces = train_forces
        self.fd_step = fd_step
        if self.train_forces:
            self.force_reg1 = MLPRegressor(hidden_layer_sizes=(50,10), activation='relu', alpha=1e-4, random_state=123, max_iter=2000)
            self.force_reg2 = MLPRegressor(hidden_layer_sizes=(40,4), activation='relu', alpha=1e-5, random_state=456, max_iter=2000)
        self.trained = False
        self._cache_atoms_id = None
        self._cache_energy = None
        self._cache_uncertainty = None

    def train(self, structures):
        X = self.descriptor.get_features(structures)[0]
        energies = np.array([atoms.get_potential_energy() for atoms in structures])
        self.reg1.fit(X, energies)
        self.reg2.fit(X, energies)
        if self.train_forces:
            forces = np.array([atoms.get_forces().flatten() for atoms in structures])
            self.force_reg1.fit(X, forces)
            self.force_reg2.fit(X, forces)
        self.trained = True

    def _predict_energy_uncertainty(self, atoms):
        atoms_id = (id(atoms), atoms.get_positions().tobytes())
        if atoms_id == self._cache_atoms_id:
            return self._cache_energy, self._cache_uncertainty
        feat = self.descriptor.create_features(atoms)
        e1 = self.reg1.predict(feat)[0]
        e2 = self.reg2.predict(feat)[0]
        mean_e = 0.5 * (e1 + e2)
        unc_e = abs(e1 - e2)
        self._cache_atoms_id = atoms_id
        self._cache_energy = mean_e
        self._cache_uncertainty = unc_e
        return mean_e, unc_e

    def predict_energy(self, atoms):
        mean_e, _ = self._predict_energy_uncertainty(atoms)
        return mean_e

    def predict_uncertainty(self, atoms):
        _, unc_e = self._predict_energy_uncertainty(atoms)
        return unc_e

    def numerical_forces_and_uncertainty(self, atoms):
        positions = atoms.get_positions()
        n_atoms = len(atoms)
        delta = self.fd_step

        displaced_atoms = []
        for i in range(n_atoms):
            for j in range(3):
                for sign in [-1, 1]:
                    atom_disp = atoms.copy()
                    pos_disp = positions.copy()
                    pos_disp[i, j] += sign * delta
                    atom_disp.set_positions(pos_disp)
                    displaced_atoms.append(atom_disp)

        X_batch = np.array([self.descriptor.create_features(a)[0] for a in displaced_atoms])

        e1_batch = self.reg1.predict(X_batch)
        e2_batch = self.reg2.predict(X_batch)

        forces = np.zeros_like(positions)
        force_unc = np.zeros_like(positions)
        idx = 0
        for i in range(n_atoms):
            for j in range(3):
                e1_m, e1_p = e1_batch[idx], e1_batch[idx+1]
                e2_m, e2_p = e2_batch[idx], e2_batch[idx+1]
                f1 = -(e1_p - e1_m) / (2 * delta)
                f2 = -(e2_p - e2_m) / (2 * delta)
                meanf = 0.5 * (f1 + f2)
                varf = abs(f1 - f2)
                forces[i, j] = meanf
                force_unc[i, j] = varf
                idx += 2
        self.force_unc = force_unc
        return forces, force_unc

    def predict_forces(self, atoms):
        if self.train_forces:
            feat = self.descriptor.create_features(atoms)
            f1 = self.force_reg1.predict(feat)[0]
            f2 = self.force_reg2.predict(feat)[0]
            avg = 0.5 * (f1 + f2)
            return avg.reshape(-1, 3)
        else:
            forces, _ = self.numerical_forces_and_uncertainty(atoms)
            return forces

    def predict_uncertainty_forces(self, atoms):
        if self.train_forces:
            feat = self.descriptor.create_features(atoms)
            f1 = self.force_reg1.predict(feat)[0]
            f2 = self.force_reg2.predict(feat)[0]
            diff = abs(f1 - f2)
            return diff.reshape(-1, 3)
        else:
            force_unc = self.force_unc
            return force_unc



class SKLEARN_RF:
    """
    Ensemble of Random Forests with uncertainty estimation (Spread).
    Default: train_forces=False (Numerical forces and uncertainty.).
    """
    def __init__(self, atoms_example, descriptor='soap', descriptor_params=None, train_forces=False, fd_step=1e-2, n_jobs=-1):
        if descriptor=='distancematrix':
            self.descriptor = ASE_DistanceMatrix_Wrapper(atoms_example)
        else:
            self.descriptor = DescriptorWrapper(atoms_example, descriptor, descriptor_params)
        self.regressor = RandomForestRegressor(n_estimators=100, n_jobs=n_jobs, random_state=42)
        self.train_forces = train_forces
        self.fd_step = fd_step
        self.n_jobs = n_jobs
        if self.train_forces:
            self.force_regressor = RandomForestRegressor(n_estimators=100, n_jobs=n_jobs, random_state=43)
        self.trained = False
        self._cache_atoms_id = None
        self._cache_energy = None
        self._cache_uncertainty = None

    def train(self, structures):
        X = self.descriptor.get_features(structures)[0]
        energies = np.array([atoms.get_potential_energy() for atoms in structures])
        self.regressor.fit(X, energies)
        if self.train_forces:
            forces = np.array([atoms.get_forces().flatten() for atoms in structures])
            self.force_regressor.fit(X, forces)
        self.trained = True

    def _predict_energy_uncertainty(self, atoms):
        atoms_id = (id(atoms), atoms.get_positions().tobytes())
        if atoms_id == self._cache_atoms_id:
            return self._cache_energy, self._cache_uncertainty
        feat = self.descriptor.create_features(atoms)
        all_preds = np.array([t.predict(feat)[0] for t in self.regressor.estimators_])
        mean_e = np.mean(all_preds)
        unc_e = np.std(all_preds)
        self._cache_atoms_id = atoms_id
        self._cache_energy = mean_e
        self._cache_uncertainty = unc_e
        return mean_e, unc_e

    def predict_energy(self, atoms):
        mean_e, _ = self._predict_energy_uncertainty(atoms)
        return mean_e

    def predict_uncertainty(self, atoms):
        try:
            unc_e=self._cache_uncertainty
        except:
            _, unc_e = self._predict_energy_uncertainty(atoms)
        return unc_e

    def numerical_forces_and_uncertainty(self, atoms):
        positions = atoms.get_positions()
        n_atoms = len(atoms)
        delta = self.fd_step

        displaced_atoms = []
        for i in range(n_atoms):
            for j in range(3):
                for sign in [-1, 1]:
                    atom_disp = atoms.copy()
                    pos_disp = positions.copy()
                    pos_disp[i, j] += sign * delta
                    atom_disp.set_positions(pos_disp)
                    displaced_atoms.append(atom_disp)

        X_batch = np.array([self.descriptor.create_features(a)[0] for a in displaced_atoms])
        all_preds = np.array([t.predict(X_batch) for t in self.regressor.estimators_])
        mean_batch = all_preds.mean(axis=0)

        forces = np.zeros_like(positions)
        force_unc = np.zeros_like(positions)
        idx = 0
        for i in range(n_atoms):
            for j in range(3):
                mean_m, mean_p = mean_batch[idx], mean_batch[idx+1]
                f = -(mean_p - mean_m) / (2 * delta)
                # Spread/Unsicherheit im Force-Diskretisierungsintervall
                std_m = all_preds[:, idx].std()
                std_p = all_preds[:, idx+1].std()
                uncertainty = 0.5 * (std_m + std_p)
                forces[i, j] = f
                force_unc[i, j] = uncertainty
                idx += 2
        self.force_unc = force_unc
        return forces, force_unc

    def predict_forces(self, atoms):
        if self.train_forces:
            feat = self.descriptor.create_features(atoms)
            all_force_preds = np.array([t.predict(feat)[0] for t in self.force_regressor.estimators_])
            meanf = np.mean(all_force_preds, axis=0)
            return meanf.reshape(-1, 3)
        else:
            forces, _ = self.numerical_forces_and_uncertainty(atoms)
            return forces

    def predict_uncertainty_forces(self, atoms):
        if self.train_forces:
            feat = self.descriptor.create_features(atoms)
            all_force_preds = np.array([t.predict(feat)[0] for t in self.force_regressor.estimators_])
            stdf = np.std(all_force_preds, axis=0)
            return stdf.reshape(-1, 3)
        else:
            force_unc = self.force_unc
            return force_unc
