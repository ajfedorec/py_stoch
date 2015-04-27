import libsbml
from mod.simulator import cuTauLeaping
import sim_maker
from scipy.integrate import odeint
from mod.utils.saver import save_results
import os


#
# GET SBML MODEL
#
# Location of your SBML model file.
sbml_file = os.getcwd() + '/plasmid_stability.xml'
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
duration = 10
# Number of points that you want to record the state of the system at. The
# points are uniformly distributed, starting from 0, across the duration of the
# simulation.
num_output_time_points = 11
# Number of times you want to run each simulation setup.
num_simulations = 256

#
# SIMULATION PARAMETERS
#   both of these parameter sets are sweepable.
#
# Initial amounts of each species. The position in the list maps to the order
# the species are listed in the given SBML file. Each position must be a list
# containing one or more values. If more than one value is given for a species,
# simulations will be run for each combination of initial values.
# e.g. [[1], [2]] or [[1, 2, 3], [4]] etc.
species_init = [[1], [0]]
# Parameter values. See description for "species_init" for details.
parameters = [[1], [1], [1], [0.99]]

#
# PARSE SETTINGS
#
# Produce a list of all combinations of simulation from the given parameters.
tl_args_list = sim_maker.TLArgsList.make_list(sbml_model, species_init,
                                              parameters, output_species,
                                              duration, num_output_time_points,
                                              num_simulations)

g_args_list = sim_maker.TLArgsList.make_list(sbml_model, species_init,
                                             parameters, output_species,
                                             duration, num_output_time_points,
                                             num_simulations, gillespie=1)

#
# RUN A SIMULATIONS
#
# TODO think about best way to store results.
tl_results = []
for args_idx, args in enumerate(tl_args_list):
    simulator = cuTauLeaping.CuTauLeaping()
    new_result = simulator.run(args)

    save_results(new_result, args, sbml_model.getName(), sim_type='TL')

    tl_results.append(new_result)

g_results = []
for args_idx, args in enumerate(g_args_list):
    simulator = cuTauLeaping.CuTauLeaping()
    new_result = simulator.run(args)

    save_results(new_result, args, sbml_model.getName(), sim_type='G')

    g_results.append(new_result)


#
# ODE COMPARISON
#
def f(y, t, args):
    Xp = y[0]
    Xm = y[1]

    Tp = args[0]
    Tm = args[1]
    lam = args[2]
    omega = args[3]

    dXp = (1 - lam) * Xp / Tp
    # dXm = (Xm / Tm) + (1 - omega) * (lam * Xp) / Tp
    dXm = (1 - omega) * (lam * Xp) / Tp
    return [dXp, dXm]

ode_results = []
for args_idx, args in enumerate(tl_args_list):
    ode_args = args.c
    y0 = args.x_0
    t = args.I
    new_result = odeint(f, y0, t, args=(ode_args,))
    save_results(new_result, args, sbml_model.getName(), sim_type='ODE')
    ode_results.append(new_result)

#####
#   RESULTS FOR SIMPLE MODEL
#
# TODO: this plotting is currently incredibly sub-standard
import matplotlib.pyplot as plt
from scipy import stats
import numpy

# plt.figure(1)
# for r_idx, result in enumerate(ode_results):
#     tl_props = tl_results[r_idx][0].astype(numpy.float32) / (tl_results[r_idx][0].astype(numpy.float32) + tl_results[r_idx][1].astype(numpy.float32))
#     g_props = g_results[r_idx][0].astype(numpy.float32) / (g_results[r_idx][0].astype(numpy.float32) + g_results[r_idx][1].astype(numpy.float32))
#     # g2_props = g2_results[r_idx][0].astype(numpy.float32) / (g2_results[r_idx][0].astype(numpy.float32) + g2_results[r_idx][1].astype(numpy.float32))
#     ode_props = ode_results[r_idx][:, 0].astype(numpy.float) / (ode_results[r_idx][:, 0].astype(numpy.float) + ode_results[r_idx][:, 1].astype(numpy.float))
#
#     mean_tl_prop = numpy.mean(tl_props, axis=1)
#     mean_g_prop = numpy.mean(g_props, axis=1)
#     # mean_g2_prop = numpy.mean(g2_props, axis=1)
#
#     my_range = numpy.linspace(0, duration, num=num_output_time_points)
#     plt.plot(my_range, mean_tl_prop, 'b:')
#     plt.plot(my_range, mean_g_prop, 'g--')
#     # plt.plot(my_range, mean_g2_prop, 'g-')
#     plt.plot(my_range, ode_props, 'r-')
#
#
#     # density_TL = stats.gaussian_kde(results[r_idx][0][1])
#     # plt_range = numpy.arange(10, 90, 0.1)
#     # plt.plot(plt_range, density_TL(plt_range), 'r-')
#     # plt.title('x_0 = ' + str(args_list[r_idx].x_0) + ', c = ' + str(args_list[r_idx].c))
# plt.title('x_0 = ' + str(species_init) + ', c = ' + str(parameters))
# plt.ylim([0, 1])
# plt.xlim([0, duration])
# plt.show()

from scipy.stats import sem

plt.figure(1)
for r_idx, result in enumerate(ode_results):
    tl_props = tl_results[r_idx][0].astype(numpy.float32) / (tl_results[r_idx][0].astype(numpy.float32) + tl_results[r_idx][1].astype(numpy.float32))
    ode_props = ode_results[r_idx][:, 0].astype(numpy.float) / (ode_results[r_idx][:, 0].astype(numpy.float) + ode_results[r_idx][:, 1].astype(numpy.float))
    g_props = g_results[r_idx][0].astype(numpy.float32) / (g_results[r_idx][0].astype(numpy.float32) + g_results[r_idx][1].astype(numpy.float32))

    median_tl_prop = numpy.median(tl_props, axis=1)
    mean_g_prop = numpy.mean(g_props, axis=1)
    median_g_prop = numpy.median(g_props, axis=1)

    mean_tl_props = []
    min_tl_props = []
    max_tl_props = []

    for time_point_samples in tl_props:
        mean_tl = numpy.mean(time_point_samples)
        mean_tl_props.append(mean_tl)
        # ci95 = 1.96 * sem(time_point_samples)
        # min_tl_props.append(mean_tl - ci95)
        # max_tl_props.append(mean_tl + ci95)

    my_range = numpy.linspace(0, duration, num=num_output_time_points)
    plt.plot(my_range, mean_tl_props, 'b--')
    plt.plot(my_range, median_tl_prop, 'b:')
    # plt.plot(my_range, min_tl_props, 'b--')
    # plt.plot(my_range, max_tl_props, 'b--')
    plt.plot(my_range, ode_props, 'r--')

    plt.plot(my_range, mean_g_prop, 'g--')
    plt.plot(my_range, median_g_prop, 'g:')

    plt.plot(my_range, tl_props, ':')

plt.title('x_0 = ' + str(species_init) + ', c = ' + str(parameters))
plt.ylim([0, 1.1])
plt.xlim([0, duration])
plt.show()

# plt.figure(1)
# for r_idx, result in enumerate(ode_results):
#     tl_props = tl_results[r_idx][0].astype(numpy.float32) / (tl_results[r_idx][0].astype(numpy.float32) + tl_results[r_idx][1].astype(numpy.float32))
#     ode_props = ode_results[r_idx][:, 0].astype(numpy.float) / (ode_results[r_idx][:, 0].astype(numpy.float) + ode_results[r_idx][:, 1].astype(numpy.float))
#
#     mean_tl_prop = numpy.mean(tl_props, axis=1)
#
#     my_range = numpy.linspace(0, duration, num=num_output_time_points)
#     plt.plot(my_range, tl_props)
#     plt.plot(my_range, mean_tl_prop, 'b:')
#     plt.plot(my_range, ode_props, 'r--')
#
# plt.title('x_0 = ' + str(species_init) + ', c = ' + str(parameters))
# plt.ylim([0, 1])
# plt.xlim([0, duration])
# plt.show()