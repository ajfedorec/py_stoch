import numpy
from parser import Parser


class GArgs:
    def __init__(self):
        # Definitions of these variables are in cuTauLeaping paper by Nobile
        # et al.
        self.c = []    # parameter values vector
        self.x_0 = []  # initial species amounts vector

        self.V = []  # flattened stoichiometry matrix

        self.M = 0  # number of reactions
        self.N = 0  # number of species
        self.V_size = 0  # number of non-zero entries in stoichiometry matrix

        self.hazards = []  # the CUDA code for UpdatePropensities

        # These should be taken from a simulation set up xml
        self.E = numpy.array([0]).astype(numpy.int32)  # indices of the output species
        self.t_max = 1  # the time at which the simulation ends
        self.ita = 11   # number of time recording points
        self.U = 128    # number of threads

        # These are derived from setup params above
        self.kappa = len(self.E)  # number of species we're recording
        self.I = numpy.array([])  # instances for recording


class GParser(Parser):
    @staticmethod
    def parse(args_out, stochastic_petri_net):
        args_out.c = numpy.array(stochastic_petri_net.c).astype(numpy.float64)

        args_out.x_0 = numpy.array(stochastic_petri_net.M).astype(numpy.int32)

        ma = stochastic_petri_net.Pre
        mb = stochastic_petri_net.Post
        mv = mb - ma

        args_out.V, args_out.V_size = GParser.flatten_matrix(mv)

        args_out.M = len(stochastic_petri_net.T)
        args_out.N = len(stochastic_petri_net.P)

        args_out.hazards = GParser.define_hazards(stochastic_petri_net)

        args_out.kappa = len(args_out.E)  # number of species we're recording
        args_out.I = numpy.linspace(0, args_out.t_max, num=args_out.ita).astype(
            numpy.float32)
        return args_out
