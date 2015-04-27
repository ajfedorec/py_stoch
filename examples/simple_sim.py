import libsbml
from mod.simulator import cuTauLeaping
import sim_maker


##
## GET SBML MODEL ##
##
sbml_file = '/home/sandy/Documents/Code/my_sim/examples/simple_sbml.xml'
reader = libsbml.SBMLReader()
document = reader.readSBML(sbml_file)
# check the SBML for errors
error_count = document.getNumErrors()
if (error_count > 0):
    raise UserWarning(str(error_count) + ' errors in SBML file: ' +
                      sbml_file)
sbml_model = document.getModel()

##
## SIMULATOR PARAMETERS
##   I don't think it would be useful for any of these parameters to be sweepable
output_species = [2]  # A list of indices of the species you want to record.
                      # Indices are 0-based in the order they are listed in the
                      # given SBML file.
duration = 0.1  # Length of the simulation in arbitrary units
num_output_time_points = 2  # Number of points that you want to record the state
                            # of the system at. The points are uniformly
                            # distributed, starting from 0, across the duration
                            # of the simulation.
num_simulations = 25600  # Number of times you want to run each simulation setup.

##
## SIMULATION PARAMETERS
##   both of these parameter sets are sweepable
#
# Initial amounts of each species. The position in the list maps to the order
# the species are listed in the given SBML file. Each position must be a list
# containing one or more values. If more than one value is given for a species,
# simulations will be run for each combination of initial values.
# e.g. [[1], [2]] or [[1, 2, 3], [4]] etc.
species_init = [[1000, 10000], [1], [0]]
# Parameter values. See description for "species_init" for details.
parameters = [[1, 2, 5], [1, 2, 5]]

## Produce a list of all combinations of simulation from the given parameters
tl_args_list = sim_maker.TLArgsList.make_list(sbml_model, species_init,
                                              parameters, output_species,
                                              duration, num_output_time_points,
                                              num_simulations)

g_args_list = sim_maker.TLArgsList.make_list(sbml_model, species_init,
                                             parameters, output_species,
                                             duration, num_output_time_points,
                                             num_simulations, gillespie=1)

##
## RUN A SIMULATIONS
##
#TODO think about best way to store results.
tl_results = []
for args_idx, args in enumerate(tl_args_list):
    simulator = cuTauLeaping.CuTauLeaping()
    tl_results.append(simulator.run(args))

g_results = []
for args_idx, args in enumerate(g_args_list):
    simulator = cuTauLeaping.CuTauLeaping()
    new_result = simulator.run(args)
    g_results.append(new_result)


#####
#   RESULTS FOR SIMPLE MODEL
#
import matplotlib.pyplot as plt
from scipy import stats
import numpy

import matplotlib.pyplot as plt
from scipy import stats
import numpy

plt.show()

plt.figure(1)
for r_idx, result in enumerate(tl_results):
    plt.subplot(6, 3, r_idx + 1)
    density_TL = stats.gaussian_kde(tl_results[r_idx][0][1])
    density_G = stats.gaussian_kde(g_results[r_idx][0][1])
    plt_range = numpy.arange(10, 90, 0.1)
    plt.plot(plt_range, density_TL(plt_range), 'r-')
    plt.plot(plt_range, density_G(plt_range), 'b-')
    plt.ylim(0, 0.1)
    plt.title('x_0 = ' + str(tl_args_list[r_idx].x_0) + ', c = ' + str(tl_args_list[r_idx].c))

plt.show()