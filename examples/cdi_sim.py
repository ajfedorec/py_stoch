import libsbml
from mod.simulator import cuTauLeaping
import sim_maker
import os
from mod.utils.saver import save_results


##
# GET SBML MODEL
##
sbml_file = os.getcwd() + '/cdi_sbml.xml'
reader = libsbml.SBMLReader()
document = reader.readSBML(sbml_file)
# check the SBML for errors
error_count = document.getNumErrors()
if error_count > 0:
    raise UserWarning(str(error_count) + ' errors in SBML file: ' +
                      sbml_file)
sbml_model = document.getModel()

#
# SIMULATOR PARAMETERS
#   I don't think it would be useful for any of these parameters to be
#   sweepable.
#
# A list of indices of the species you want to record. Indices are 0-based in
# the order they are listed in the given SBML file.
output_species = [0, 1]
# Length of the simulation in arbitrary units.
duration = 5
# Number of points that you want to record the state of the system at. The
# points are uniformly distributed, starting from 0, across the duration of the
# simulation.
num_output_time_points = 100
# Number of times you want to run each simulation setup.
num_simulations = 32

#
# SIMULATION PARAMETERS
#   both of these parameter sets are sweepable
#
# Initial amounts of each species. The position in the list maps to the order
# the species are listed in the given SBML file. Each position must be a list
# containing one or more values. If more than one value is given for a species,
# simulations will be run for each combination of initial values.
# e.g. [[1], [2]] or [[1, 2, 3], [4]] etc.
species_init = [[10], [10]]
# Parameter values. See description for "species_init" for details.
parameters = [[0.5, 1], [1], [0.001, 0.01, 0.1]]

# Produce a list of all combinations of simulation from the given parameters
tl_args_list = sim_maker.TLArgsList.make_list(sbml_model, species_init,
                                              parameters, output_species,
                                              duration, num_output_time_points,
                                              num_simulations)

# g_args_list = sim_maker.TLArgsList.make_list(sbml_model, species_init,
#                                              parameters, output_species,
#                                              duration, num_output_time_points,
#                                              num_simulations, gillespie=1)

#
# RUN A SIMULATIONS
#
tl_results = []
for args_idx, args in enumerate(tl_args_list):
    simulator = cuTauLeaping.CuTauLeaping()
    new_result = simulator.run(args)
    tl_results.append(new_result)
    save_results(new_result, args, sbml_model.getName(), sim_type='TL')

# g_results = []
# for args_idx, args in enumerate(g_args_list):
#     simulator = cuTauLeaping.CuTauLeaping()
#     new_result = simulator.run(args)
#     g_results.append(new_result)
    # save_results(new_result, args, sbml_model.getName(), sim_type='G')


# #####
# #   RESULTS FOR SIMPLE MODEL
# #
# import matplotlib.pyplot as plt
# from scipy import stats
# import numpy
# simR_TL = tl_results[1]
# # simR_G = g_results[0]
#
# my_range = numpy.linspace(0, duration, num=num_output_time_points)
#
# mean_TL_0 = numpy.mean(simR_TL[0], axis=1)
# mean_TL_1 = numpy.mean(simR_TL[1], axis=1)
#
# sd_TL_0 = numpy.std(simR_TL[0], axis=1)
# sd_TL_1 = numpy.std(simR_TL[1], axis=1)
#
# # mean_G_0 = numpy.mean(simR_G[0], axis=1)
# # mean_G_1 = numpy.mean(simR_G[1], axis=1)
# #
# # sd_G_0 = numpy.std(simR_G[0], axis=1)
# # sd_G_1 = numpy.std(simR_G[1], axis=1)
#
# plt.figure(1)
# plt.subplot(211)
# plt.xlabel('Time')
# plt.ylabel('Mean')
# plt.title('Variation of mean')
# plt.plot(my_range, mean_TL_0, 'b-')
# plt.plot(my_range, mean_TL_1, 'g-')
#
# # plt.plot(my_range, mean_G_0, 'r-')
# # plt.plot(my_range, mean_G_1, 'y-')
#
# plt.subplot(212)
# plt.xlabel('Time')
# plt.ylabel('Standard deviation')
# plt.title('Variation of standard deviation')
# plt.plot(my_range, sd_TL_0, 'b-')
# plt.plot(my_range, sd_TL_1, 'g-')
#
# # plt.plot(my_range, sd_G_0, 'r-')
# # plt.plot(my_range, sd_G_1, 'y-')
# plt.show()
#
# plt.figure(1)
# for r_idx, result in enumerate(tl_results):
#     plt.subplot(2, 3, r_idx + 1)
#     density_TL1 = stats.gaussian_kde(tl_results[r_idx][0][num_output_time_points-1])
#     # density_TL2 = stats.gaussian_kde(tl_results[r_idx][1][num_output_time_points-1])
#     plt_range = numpy.arange(0, 1000, 0.1)
#     plt.plot(plt_range, density_TL1(plt_range), 'b-')
#     # plt.plot(plt_range, density_TL2(plt_range), 'g-')
#     # plt.ylim(0, 0.1)
#     plt.title('x_0 = ' + str(tl_args_list[r_idx].x_0) + ', c = ' +
#               str(tl_args_list[r_idx].c))
#
# plt.show()
