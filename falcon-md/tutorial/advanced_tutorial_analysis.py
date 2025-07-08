""" Options for plotting your results after the advanced tutorial OTF-MD simulation"""


# Generates plot of the energy versus time and highlights OTF training steps during the simulation.
from falcon-md.utils.analysis.simulation import plot_simulation

plot_simulation("MD.traj", "INSERT_FALCON_OUTPUTFILE.out")



# Generates plot of the radial distribution function averaged over two different time frames of the MD simulation.

from falcon-md.utils.analysis.rdf import plot_rdf

plot_rdf(filename="MD.traj",
         index_solid="3000:5000:200",         # For smoother RDF's you need to average over more structures/a longer timeframe. However, this will increase plotting time.
         index_liquid="20000:22000:200")

