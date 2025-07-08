from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write,Trajectory
from sklearn.cluster import KMeans
import numpy as np
import time
import copy

class FALCON(Calculator):
    """
    On-the-Fly Machine Learning Calculator

    Parameters
    -----------
    model : falcon-md.models.model_base_class.ModelBaseClass
        ML model to use for the OTF training.
    
    calc : ASE calculator
        ASE calculator object to perform exact calculations.

    training_data : list of ASE atoms objects
        List of structures for initial training of the ML model.

    accuracy_e : float
        Accuracy threshold (Epsilon) in eV. Default: 0.1

    accuracy_f : float, optional
        Optional accuracy threshold for forces in eV/Å. Default: inf

    modelsize : int
        Average model size, defines number of structures before a new ML model is added. Default: 500

    max_clusters : int
        Maximum of clusters/ML models. Default: 1
    
    train_start : int
        Number of initial predictions, where the ML model is trained regardless of the uncertainty. Default: 10

    train_every : int
        Periodic retraining interval, regardless of uncertainty. Default: 1e9

    train_log : bool
        If True, retraining occurs at exponentially increasing intervals (2**x steps). Default: True.

    write_training_data : bool
        Save training structures to file during simulation. Default: True.

    td_filename : str
        Filename for training data output. Default: "OTF_Training_Data.traj".

    """

    
    implemented_properties = ['energy', 'forces']
    default_parameters = {}



    def __init__(self,
                 model,
                 calc,
                 training_data, 
                 accuracy_e=0.1, 
                 accuracy_f=float('inf'), 
                 modelsize=500, 
                 max_clusters=1, 
                 train_start=10, 
                 train_every=1e9, 
                 train_log=True, 
                 write_training_data=True,  
                 td_filename = 'OTF_Training_Data.traj', 
                 **kwargs):

        Calculator.__init__(self, **kwargs)

        self.calc = calc
        self.steps = 0               # MD step count 
        self.training_steps = 0      # Count of performed retrainings.
        self.initialize = True
        self.accuracy_e = accuracy_e
        self.accuracy_f = accuracy_f
        self.training_data = training_data
        self.train_start = train_start
        self.train_every = train_every
        self.write_training_data = training_data
        self.training_data_filename = td_filename 
        self.modelsize = modelsize
        self.log_test = (lambda x: (np.log2(x) % 1) == 0) if train_log else (lambda x: False)

        # Setup of the ML models
        self.models = []
        self.active_models = []
        self.descriptor = model.descriptor
        self.last_clustering = 0
        self.n_clusters = 0
        self.max_clusters = max_clusters

        # Creation of number of  ML models that is defined with max_clusters.
        for i in range(max_clusters):
            model_copy = copy.deepcopy(model)
            setattr(self, f"model{i}", model_copy)
            self.models.append(getattr(self, f"model{i}"))




    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=['positions']):
        Calculator.calculate(self, atoms, properties, system_changes)

        print(f'\n============================ Step: {self.steps+1} ============================\n')
    
        print(f'Total structures in training data: {len(self.training_data)}')


        # Clustering of training data into subsets of data (triggered if modelsize is reached or self.initialize == True)
        if ((len(self.training_data) % self.modelsize == 0 and self.n_clusters <= self.max_clusters) or self.initialize) and len(self.training_data) != self.last_clustering:
            self.last_clustering = len(self.training_data)

            if self.n_clusters < self.max_clusters:
                start=time.time()
                
                self.n_clusters +=1
                new_model = self.models.pop(0)
                self.active_models.append(new_model)
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
                    model.train(self.clusters[i])
                    end_model = time.time()
                    t_model = end_model - start_model
                    print(f"Model {i} trained with {len(self.clusters[i])} structures ({t_model} s).")
                    print()
                end = time.time()
                t = end-start
                if self.initialize:
                    if self.write_training_data:
                        with Trajectory(self.training_data_filename, mode='w') as traj:
                            for structure in self.training_data:
                                traj.write(structure)
                    print(f'Model trained with initial training data. ({t} s)')
                    self.initialize = False

                else:
                    print(f'Step {self.steps}: Clustering completed. {len(self.active_models)} models trained. ({t} s)')
                print()
                

        # Prediction of energy, forces and uncertainties
        E = None
        F = None
        Fmax = None
        Fstdmax = float('inf')
        best_model = None
        Estd = float('inf')
        for i, model in enumerate(self.active_models):
            print(f'Model {i}...')
            start = time.time()
            E_model = model.predict_energy(atoms)
            end = time.time()
            t = end-start
            print(f'Predict energy ({t} s)')

            start = time.time()
            Estd_model = model.predict_uncertainty(atoms)
            end = time.time()
            t = end-start
            print(f'Predict energy_uncertainty ({t} s)')
            start = time.time()
            F_model = model.predict_forces(atoms)
            Fmax_model = np.max(np.linalg.norm(F_model, axis=1))
            end = time.time()
            t = end-start
            print(f'Predict forces ({t} s)')

            start = time.time()
            Fstd_model = model.predict_uncertainty_forces(atoms)
            Fstdmax_model = np.max(np.linalg.norm(Fstd_model, axis=1))
            end = time.time()
            t = end-start
            print(f'Predict forces_uncertainty ({t} s)')


            print(f'E = {E_model} (Model {i})')
            print(f'Estd = {Estd_model} (Model {i})')
            print(f'F = {Fmax_model} (Model {i})')
            print(f'Fstd = {Fstdmax_model} (Model {i})')
            print()

             # Select model with lowest uncertainty
            if Estd_model<Estd:
                Estd = Estd_model
                E = E_model
                F = F_model
                Fmax = Fmax_model
                Fstdmax = Fstdmax_model
                best_model = i

        print(f'Model {best_model} has the lowest uncertainty.')
        print(f'E = {E}')
        print(f'Estd = {Estd}')
        print(f'Fmax = {Fmax}')
        print(f'Fstd = {Fstdmax}')
        print()
        
        # Determine if an exact calculation is necessary and perform it
        if self.steps<self.train_start or self.steps % self.train_every == 0 or self.log_test(self.steps) or Estd>self.accuracy_e or Fstdmax>self.accuracy_f:
           print('Exact calculation...')           
           b=atoms.copy()
           b.calc=copy.deepcopy(self.calc)
           start = time.time()
           E=b.get_potential_energy()
           F=b.get_forces()
           end = time.time()
           t = end-start
           self.steps += 1
           print(f'Exact Calculation ({t} s)')
           print(f'Real energy = {E}')
           print(f'Real Fmax = {np.max(np.linalg.norm(F, axis=1))}')

           self.training_data.append(b)

           # Append new training structure to OTF training data file.
           if self.write_training_data:
                with Trajectory(self.training_data_filename, mode='a') as traj:
                    traj.write(b)

           
           start = time.time()

           # Predict nearest cluster of new training structure
           Y = self.descriptor.create_features(b)
           pred_cluster = self.kmeans.predict(Y)
           print(f'New structure is added to Model {pred_cluster[0]}.')
           print()
           self.clusters[pred_cluster[0]].append(b)

           # Only retrain the updated ML model
           self.active_models[pred_cluster[0]].train(self.clusters[pred_cluster[0]])
           end = time.time()
           t = end-start
           self.training_steps += 1
           print(f'Step {self.steps}: Model {pred_cluster[0]} trained with {len(self.clusters[pred_cluster[0]])} Structures ({t} s). ({self.training_steps} Training steps.) ')
           print()


        else:
           self.steps += 1
           print(f'Step {self.steps}: Model not trained.')
           print()
        self.results['forces'] = F
        self.results['energy'] = E
        

    # Function to get features of an ASE atoms object
    def get_features(self, structures):
        features = np.array(self.descriptor.get_features(structures)).sum(axis=1)
        return features
