from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write,Trajectory
from sklearn.cluster import KMeans
import numpy as np
import time
import copy

class FALCON_Static(Calculator):
    """
    Machine Learning Calculator using a static model.

    Parameters
    -----------
    model : falcon-md.models.model_base_class.ModelBaseClass or list of Models
        ML model to use or list of models to load.
 
    training_data : list of ASE atoms objects
        List of structures for training of the ML model.

    n_clusters : int
        Number of clusters/ML models used for training. Default: 1

    """

    implemented_properties = ['energy', 'forces']
    default_parameters = {}

    def __init__(self,
                 model,
                 training_data, 
                 n_clusters=1,
                 **kwargs):

        Calculator.__init__(self, **kwargs)

        self.steps = 0               # MD step count 
        self.initialize = True
        self.training_data = training_data
        self.n_clusters = n_clusters

        # Setup of the ML models
        if not type(model)==list:
            self.active_models = []
            self.descriptor = model.descriptor
            self.reloaded = False

        # Creation of number of  ML models that is defined with n_clusters.
            for i in range(n_clusters):
                model_copy = copy.deepcopy(model)
                setattr(self, f"model{i}", model_copy)
                self.active_models.append(getattr(self, f"model{i}"))

        else:
            self.models = model
            self.active_models = model
            self.descriptor = model[0].descriptor
            self.n_clusters = len(model)
            self.reloaded = True



    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=['positions']):
        Calculator.calculate(self, atoms, properties, system_changes)

        print(f'\n============================ Step: {self.steps+1} ============================\n')
    
        print(f'Total structures in training data: {len(self.training_data)}')

        if self.initialize:
            start = time.time()

            print(f'{len(self.active_models)} active Models.')

            # Cluster all structures in training data using KMeans
            X = self.get_features(self.training_data)
            self.kmeans = KMeans(n_clusters=self.n_clusters, init="k-means++", n_init=10, random_state=np.random.randint(0, 10e6)).fit(X)
            self.labels = self.kmeans.labels_
            print(self.labels)
            self.clusters = {i: [] for i in range(self.n_clusters)}

            # Assign structures to their ML model.
            for structure, label in zip(self.training_data, self.labels):
                self.clusters[label].append(structure)

            for cluster_id, structures in self.clusters.items():
                print(f"Cluster {cluster_id} contains {len(structures)} objects.")
            end = time.time()
            t = end-start
            print(f"Clustering of {len(self.training_data)} took {t} s.")

            # Training of all active ML models.
            start = time.time()
            for i, model in enumerate(self.active_models):
                start_model = time.time()
                print(type(model))
                model.train(self.clusters[i])
                end_model = time.time()
                t_model = end_model - start_model
                print(f"Model {i} trained with {len(self.clusters[i])} structures ({t_model} s).")
                print()
            end = time.time()
            t = end-start
            print(f'Model trained with initial training data. ({t} s)')
            self.initialize = False

        # Prediction of energy, forces and uncertainties
        E = None
        F = None
        Fmax = None

        for i, model in enumerate(self.active_models):
            print(f'Model {i}...')
            start = time.time()
            E_model = model.predict_energy(atoms)
            end = time.time()
            t = end-start
            print(f'Predict energy ({t} s)')
            print(f'E = {E_model} (Model {i})')

            start = time.time()
            F_model = model.predict_forces(atoms)
            Fmax_model = np.max(np.linalg.norm(F_model, axis=1))
            end = time.time()
            t = end-start
            print(f'Predict forces ({t} s)')
            print(f'F = {Fmax_model} (Model {i})')
            print()

        if self.n_clusters>1:
            best_model = None
            Estd = float('inf')
            for i, model in enumerate(self.active_models):
                start = time.time()
                Estd_model = model.predict_uncertainty(atoms)
                end = time.time()
                t = end-start
                print(f'Predict energy_uncertainty ({t} s)')
                print(f'Estd = {Estd_model} (Model {i})')
                if Estd_model<Estd:
                    Estd = Estd_model
                    E = E_model
                    F = F_model
                    Fmax = Fmax_model
                    Fstdmax = Fstdmax_model
                    best_model = i
            print()
            print(f'Model {best_model} has the lowest uncertainty.')
            print(f'E = {E}')
            print(f'Estd = {Estd}')
            print(f'Fmax = {Fmax}')
            print(f'Fstd = {Fstdmax}')
            print()

        else:
            E = E_model
            F = F_model
        
        self.steps += 1
        print(f'Step {self.steps}: Model not trained.')
        print()
        self.results['forces'] = F
        self.results['energy'] = E
        

    # Function to get features of an ASE atoms object
    def get_features(self, structures):
        features = np.array(self.descriptor.get_features(structures)).sum(axis=1)
        return features


