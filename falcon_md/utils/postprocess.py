import os
import copy
from ase.io import read, write, Trajectory
from sklearn.cluster import KMeans
import numpy as np

def get_uncertain_points(model, trajectory, n_clusters=50, points_per_cluster=1, threshold=1e9, write_structures=True, path='Uncertain_Strcutures.traj'):
    traj = read(trajectory, index=':')


    uncertainties=[]

    for i, atoms in enumerate(traj):
        best_unc = float('inf')
        for i, m in enumerate(model):
            unc = m.predict_uncertainty(atoms)
            if unc < best_unc:
                best_unc = unc
        uncertainties.append(best_unc)

    descriptor = model[0].descriptor
    X = np.array(descriptor.get_features(traj)).sum(axis=1)
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=np.random.randint(0, 10e6)).fit(X)
    labels = kmeans.labels_
    clusters = [[] for _ in range(n_clusters)]
    cluster_unc = [[] for _ in range(n_clusters)]
    idx = [[] for _ in range(n_clusters)]

    for i, (structure, label, uncertainty) in enumerate(zip(traj, labels, uncertainties)):
        clusters[label].append(structure)
        cluster_unc[label].append(uncertainty)
        idx[label].append(i)

    uncertain_structures = []

    print()
    print(f"Most uncertain points in the analyzed Trajectory:")
    for cluster_idx in range(len(clusters)):
        unc_values = cluster_unc[cluster_idx]
        sorted_indices = sorted(range(len(unc_values)), key=lambda i: unc_values[i], reverse=True)
        top_n_indices = sorted_indices[:points_per_cluster]


        for i in top_n_indices:
            if cluster_unc[cluster_idx][i]>threshold:
                print(f"MD step: {idx[cluster_idx][i]},  Uncertainty: {cluster_unc[cluster_idx][i]} eV, Cluster: {cluster_idx}")
                uncertain_structures.append(clusters[cluster_idx][i])

    if write_structures:
        with Trajectory(path, mode='w') as traj:
                            for structure in uncertain_structures:
                                traj.write(structure)
