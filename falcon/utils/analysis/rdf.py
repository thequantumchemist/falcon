import numpy as np
import matplotlib.pyplot as plt
from ase.geometry.analysis import Analysis
from ase.io import read
from ase.build import make_supercell


def plot_rdf(filename: str,
             index_solid: str,
             index_liquid: str,
             rmax: float = 8.0,
             nbins: int = 200):
    """
    Compute and plot average RDFs for solid and liquid segments of a trajectory.

    Parameters
    ----------
    filename : str
        Path to trajectory file.
    index_solid : str
        ASE slice string for solid frames (e.g. '4000:5000:200').
    index_liquid : str
        ASE slice string for liquid frames (e.g. '49000:50000:200').
    rmax : float, optional
        Maximum distance in Å for RDF calculation (default: 8.0).
    nbins : int, optional
        Number of RDF bins (default: 200).
    """

    def compute_avg_rdf(traj):
        rdf_accum = []
        for atoms in traj:
            sc = make_supercell(atoms, np.identity(3) * 3)
            rdf = Analysis(sc).get_rdf(rmax=rmax, nbins=nbins)[0]
            rdf_accum.append(rdf)
        return np.mean(rdf_accum, axis=0)

    traj_solid = read(filename, index=index_solid)
    traj_liquid = read(filename, index=index_liquid)

    r = np.linspace(0.0, rmax, nbins)
    rdf_solid = compute_avg_rdf(traj_solid)
    rdf_liquid = compute_avg_rdf(traj_liquid)

    colors = ["#CCBB44", "#EE6677"]
    plt.figure(figsize=(5, 5))
    plt.plot(r, rdf_solid, color=colors[0], linewidth=2, label='Solid')
    plt.plot(r, rdf_liquid, color=colors[1], linewidth=2, label='Liquid')

    plt.ylabel('g(r)')
    plt.xlabel(r'r ($\mathrm{\AA}$)')
    plt.yticks([])
    xticks_major = np.arange(1, rmax, 1)
    xticks_minor = np.arange(1.5, rmax, 1)

    plt.xticks(xticks_major)
    plt.xlim(1, rmax)

    ax = plt.gca()
    ax.set_xticks(xticks_minor, minor=True)

    plt.legend(frameon=True, fontsize=12, markerscale=2/3)
    plt.tight_layout()
    plt.show()
