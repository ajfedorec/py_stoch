import numpy
import string

from jinja2 import Template

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
        self.U = 12800


class GParser:
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

    @staticmethod
    def define_hazards(spn):
        hazards_mod = Template(
            """
            __device__ void UpdatePropensities(float* a, uint* x, float* c)
            {
            {{hazards}}
            }
            """)
        hazards_list_string = ''
        for haz_idx, haz in enumerate(spn.h):
            hazards_list_string = hazards_list_string + '    ' + 'a[' + str(
                haz_idx) + '] = ' + haz + ';\n'
        hazards_mod_code = hazards_mod.render(hazards=hazards_list_string)
        for i in range(len(spn.P), 0, -1):
            hazards_mod_code = string.replace(hazards_mod_code,
                                              'species_' + str(i - 1),
                                              'x[' + str(i - 1) + ']')
        for i in range(len(spn.c), 0, -1):
            hazards_mod_code = string.replace(hazards_mod_code,
                                              'parameter_' + str(i - 1),
                                              'c[' + str(i - 1) + ']')

        # mymod = SourceModule(hazards_mod_code)
        # print hazards_mod_code
        return hazards_mod_code

    @staticmethod
    def flatten_matrix(matrix):
        # each entry in the flattened matrix is an uchar4
        f_matrix = []

        count = 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i][j] != 0:
                    entry = numpy.array(
                        [numpy.byte(i), numpy.byte(j), numpy.byte(matrix[i][j]),
                         numpy.byte(0)])
                    f_matrix.append(entry)
                    count += 1
        return numpy.array(f_matrix), numpy.ubyte(count)
