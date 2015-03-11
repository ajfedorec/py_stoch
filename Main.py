import libsbml
import getopt
import sys
from datetime import datetime

from mod.simulator import cuTauLeaping, cuGillespie
import pycuda.autoinit


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "m:s:t:", ["model=", "simulation=", "type="])
    except getopt.GetoptError:
        # usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-m", "--model"):
            # READ IN SBML
            # in:  path to sbml file
            # out: sbml_model object
            sbml_file = arg
            reader = libsbml.SBMLReader()
            document = reader.readSBML(sbml_file)
            # check the SBML for errors
            error_count = document.getNumErrors()
            if (error_count > 0):
                raise UserWarning(str(error_count) + ' errors in SBML file: ' +
                                  sbml_file)
            sbml_model = document.getModel()

        elif opt in ("-s", "--simulation"):
            settings_file = arg
        elif opt in ("-t", "--type"):
            sim_type = arg

    # INITIATE SIMULATION
    # in:  simulationArguments object, "simulation setup info"?
    #   out: simulation exit message
    if sim_type in "TL":
        simulator = cuTauLeaping.CuTauLeaping()
    elif sim_type in "G":
        simulator = cuGillespie.CuGillespie()
    sim_result = simulator.run(sbml_model, settings_file)

    # # WRITE OUT RESULTS
    # now = datetime.now()
    # dt = str(now.year) + '-' + str(now.month) + '-' + str(now.day) + '-' + str(now.hour)+ '-' + str(now.minute)
    # for i in range(len(sim_result)):
    #     filename = 'var' + str(i) + '_' + dt
    #     if out_dir is not None:
    #         filename = "{0}/{1}".format(out_dir, filename)
    #     numpy.savetxt(filename, sim_result[i], fmt='%1u')

    return sim_result

##########
# TEST
##########
# # sbml_file = '/home/sandy/Documents/Code/my_sim/examples/lotkaVoltera_sbml.xml'
# sbml_file = '/home/sandy/Documents/Code/my_sim/examples/schlogl_sbml.xml'
# # sbml_file = '/home/sandy/Documents/Code/my_sim/examples/simple_sbml.xml'
#
# # set_file = '/home/sandy/Documents/Code/my_sim/examples/lotkaVolterra_settings.xml'
# set_file = '/home/sandy/Documents/Code/my_sim/examples/schlogl_settings.xml'
# # set_file = '/home/sandy/Documents/Code/my_sim/examples/simple_settings.xml'
#
# main(['-m', sbml_file, '-s', set_file])
#
# pycuda.driver.stop_profiler()