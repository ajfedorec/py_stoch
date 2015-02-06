import libsbml
import getopt
import numpy
import sys
from datetime import datetime

from simulator import cuTauLeaping, cuGillespie


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "m:s:", ["model=", "simulation="])
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
                raise UserWarning(error_count + ' errors in SBML file: ' +
                                  open_file_.name)
            sbml_model = document.getModel()

        #elif opt in ("-s", "--simulation"):
            # READ IN SIM_XML
            # in:  path to simulation xml file
            # out: sim_info object
            # TODO




    # INITIATE SIMULATION
    # in:  simulationArguments object, "simulation setup info"?
    #   out: simulation exit message
    # if sim_type in "TL":
    #     simulator = cuTauLeaping.CuTauLeaping()
    # elif sim_type in "G":
    #     simulator = cuGillespie.CuGillespie()
    # sim_result = simulator.run(sbml_model)

    # print sim_result

    sim1 = cuTauLeaping.CuTauLeaping()
    sim2 = cuGillespie.CuGillespie()
    simR1 = sim1.run(sbml_model)
    simR2 = sim2.run(sbml_model)

    import matplotlib.pyplot as plt
    from scipy import stats

    density1 = stats.gaussian_kde(simR1[2][1])
    density2 = stats.gaussian_kde(simR2[2][1])

    plt_range = numpy.arange(10, 90, 1)

    plt.plot(plt_range, density1(plt_range), 'r-')
    plt.plot(plt_range, density2(plt_range), 'b-')
    # plt.axis([10, 90, 0, 0.07])
    plt.show()

    # # WRITE OUT RESULTS
    # now = datetime.now()
    # dt = str(now.year) + '-' + str(now.month) + '-' + str(now.day) + '-' + str(now.hour)+ '-' + str(now.minute)
    # for i in range(len(sim_result)):
    #     filename = 'var' + str(i) + '_' + dt
    #     if out_dir is not None:
    #         filename = "{0}/{1}".format(out_dir, filename)
    #     numpy.savetxt(filename, sim_result[i], fmt='%1u')

    # # CHECK SIMULATION RESULTS
    # if sim_result == 0:
    #     sys.exit(0)
    # else:
    #     # TODO
    #     sys.exit(1)


# if __name__ == "__main__":
#     main(sys.argv[1:])

##########
# TEST
##########
sim_type = "TL"
out_dir = None

sbml_file = '/home/sandy/Downloads/simple_sbml.xml'
main(['-m', sbml_file])