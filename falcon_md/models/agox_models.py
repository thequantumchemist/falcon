"""Defines three default models implemented AGOX, that can be imported from falcon_md.models.agox_models"""

def GPR(atoms):
    """Standard Gaussian Progess Regression Model trained only on energies"""
    from agox.models.descriptors import Fingerprint
    from agox.models.GPR import GPR
    from agox.models.GPR.kernels import RBF, Noise
    from agox.models.GPR.kernels import Constant as C
    from agox.models.GPR.priors import Repulsive

    descriptor = Fingerprint.from_atoms(atoms)
    beta = 0.01
    k0 = C(beta, (beta, beta)) * RBF()
    k1 = C(1 - beta, (1 - beta, 1 - beta)) * RBF()
    kernel = C(5000, (1, 1e5)) * (k0 + k1) + Noise(0.01, (0.01, 0.01))

    model = GPR(
        descriptor=descriptor,
        kernel=kernel,
        prior=Repulsive(),
        use_ray=True
    )

    print("""\nYou are using the Gaussian Process Regression model implemented in AGOX. If you use this model in a publication, please cite:
Mads-Peter V. Christiansen, Nikolaj Rønne, Bjørk Hammer, "Atomistic Global Optimization X: A Python package for optimization of atomistic structures", J. Chem. Phys. 157, 054701 (2022),
<https://arxiv.org/abs/2204.01451>\n""")

    return model


def GPRForces(atoms):
    """Gaussian Progess Regression Model trained on forces and energies"""
    from agox.models.descriptors import Fingerprint
    from agox.models.GPR import SparseGPR
    from agox.models.GPR.kernels import RBF
    from agox.models.GPR.kernels import Constant as C

    descriptor = Fingerprint.from_atoms(atoms)
    kernel = C(1, (1, 100)) * RBF(20, (10, 30))

    model = SparseGPR(
        kernel=kernel,
        descriptor=descriptor,
        sparsifier=None,
        noise_E=0.01,  # eV/atom
        noise_F=0.05,  # eV/Å/atom
        force_data_filter="all",
        train_uncertainty=True,
        use_ray=True
    )
    
    print("""\nYou are using the Gaussian Process Regression model implemented in AGOX. If you use this model in a publication, please cite:
Mads-Peter V. Christiansen, Nikolaj Rønne, Bjørk Hammer, "Atomistic Global Optimization X: A Python package for optimization of atomistic structures", J. Chem. Phys. 157, 054701 (2022),
<https://arxiv.org/abs/2204.01451>\n""")

    return model


def SparseGPR(atoms):
    """Sparse Gaussian Progess Regression Model trained on forces and energies"""
    from agox.models.descriptors import Fingerprint
    from agox.models.GPR import SparseGPR
    from agox.models.GPR.kernels import RBF
    from agox.models.GPR.kernels import Constant as C
    from agox.utils.sparsifiers import CUR

    descriptor = Fingerprint.from_atoms(atoms)
    kernel = C(1, (1, 100)) * RBF(20, (10, 30))

    model = SparseGPR(
        kernel=kernel,
        descriptor=descriptor,
        sparsifier=CUR(1000),  # CUR-Sparsifizierung mit 1000 Punkten
        noise_E=0.01,  # eV/atom
        noise_F=0.05,  # eV/Å/atom
        force_data_filter="all",
        train_uncertainty=True,
        use_ray=True
    )
    
    print("""\nYou are using the SparseGPR model implemented in AGOX. If you use this model in a publication, please cite:
Mads-Peter V. Christiansen, Nikolaj Rønne, Bjørk Hammer, "Atomistic Global Optimization X: A Python package for optimization of atomistic structures", J. Chem. Phys. 157, 054701 (2022),
<https://arxiv.org/abs/2204.01451>\n""")

    return model
