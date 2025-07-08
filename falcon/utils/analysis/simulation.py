import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import re

def plot_simulation(traj, outfile, timestep_fs=0.5, max_steps=None):
    """
    Plot potential energy over time with DFT-call markers from outfile.

    Parameters
    ----------
    traj : str
        Path to ASE trajectory file.

    outfile : str
        Path to the FALCON outputfile containing training events.

    timestep_fs : float
        Time per MD step in fs.

    max_steps : int, optional
        Limit number of trajectory steps to read. Default: None (read all).
    """

    # Read trajectory
    traj = read(traj, index=f':{max_steps}' if max_steps else ':')
    energies = [atoms.get_potential_energy() for atoms in traj]
    times_ps = np.arange(len(energies)) * (timestep_fs / 1000)  # Convert fs to ps

    # Parse outputfile for DFT call steps (MD steps where training occurred)
    pattern = r"Step\s+(\d+):\s+Model\s+\d+\s+trained\s+with\s+\d+\s+Structures\s+\([\d.]+ s\).\s+\((\d+)\s"
    with open(outfile, 'r') as f:
        text = f.read()

    matches = re.findall(pattern, text)
    dft_md_steps = [int(md_step) for md_step, _ in matches if int(md_step) < len(energies)]
    dft_md_steps = np.array(dft_md_steps)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(times_ps, energies, linewidth=2, alpha=0.6, color='#E41A1C', label="Energy")
    plt.scatter(times_ps[dft_md_steps], np.array(energies)[dft_md_steps],
                color='k', s=6, label='DFT call', zorder=5)

    plt.xlabel('Time (ps)')
    plt.ylabel('Energy (eV)')
    plt.legend(handlelength=0.1, markerscale=2, frameon=False)
    plt.tight_layout()
    plt.show()
