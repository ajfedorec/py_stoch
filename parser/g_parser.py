import numpy

from parser import Parser
from petri_net import SPN


class GArgs:
    def __init__(self):
        # definitions of these variables are in cuTauLeaping paper by Nobile
        # et al.
        self.c = []
        self.x_0 = []

        self.V = []

        self.M = 0
        self.N = 0
        self.V_size = 0

        self.hazards = []

        # these should be taken from a simulation set up xml
        self.E = numpy.array([0, 1, 2]).astype(
            numpy.uint8)  # indices of the output species
        self.ita = 2  # number of time recording points
        self.kappa = len(self.E)  # number of species we're recording
        self.t_max = 0.1  # the time at which the simulation ends

        self.I = numpy.array([])  # instances for recording

        # self.U = 998400
        self.U = 1280000


class GParser(Parser):
    @staticmethod
    def parse(sbml_model):
        # THESE ARE TAKEN FROM THE SBML_MODEL
        stochastic_petri_net = SPN()
        stochastic_petri_net.sbml_2_stochastic_petri_net(sbml_model)

        args_out = GArgs()

        args_out.c = numpy.array(stochastic_petri_net.c).astype(numpy.float32)

        args_out.x_0 = numpy.array(stochastic_petri_net.M).astype(numpy.uint32)

        ma = stochastic_petri_net.Pre
        mb = stochastic_petri_net.Post
        mv = mb - ma

        args_out.V, args_out.V_size = GParser.flatten_matrix(mv)

        args_out.M = len(stochastic_petri_net.T)
        args_out.N = len(stochastic_petri_net.P)

        args_out.hazards = GParser.define_hazards(stochastic_petri_net)

        # THESE ARE TAKEN FROM THE SIMULATION XML
        # TODO
        # args_out.ita = 0
        # args_out.kappa = 0
        #
        # args_out.t_max = 0
        #
        # args_out.E = []

        args_out.I = numpy.linspace(0, args_out.t_max, num=args_out.ita).astype(
            numpy.float32)
        return args_out
