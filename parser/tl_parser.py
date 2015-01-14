# VARIABLE | DESCRIPTION                       | TYPE  | DIMENSIONS | KERNEL
#
# ********** MODEL PARAMETERS **********
# c        - stochastic constants              - float - M          - P1-P2, P3 //
# x_0      - initial system state              - uint  - N
#
# ********** THE MODEL **********
# A        - flattened Pre matrix              - uchar4 - A_size      //
# V        - flattened stoichiometry matrix    - uchar4 - V_size      //  CONSTANT
# V_t      - flat transpose of stoich matrix   - uchar4 - V_t_size    //  MEMORY
# V_bar    - flat constant stoich matrix       - uchar4 - V_bar_size  //
#
# hazards  - function of hazard functions      - function
#
# ********** ATTRIBUTES OF THE MODEL **********
# ita      - number of time instants for output- uint  - 1            //
# kappa    - number of output species          - uint  - 1            //
# M        - number of reactions               - uint  - 1            //
# N        - number of species                 - uint  - 1            //  CONSTANT
# A_size   -                                   - uint  - 1            //  MEMORY
# V_size   -                                   - uint  - 1            //
# V_t_size -                                   - uint  - 1            //
# V_bar_size  -                                - uint  - 1            //
#
# H        - HOR for each species              - uint  - N            //  COULD BE
# H_type   - stoichiometry of the HOR          - uint  - N            //  CONSTANT?
#
# ********** PARAMETERS OF THE SIMULATION **********
# n_c      - critical reaction threshold       - uint  - 1            //  CONSTANT
# eta      - error control parameter           - uint  - 1            //  MEMORY
# t_max    - simulation length                 - float?- 1            //
#
# E    - indices of the output species         - uint  - kappa            //

import string
import re
import numpy

import sympy
from jinja2 import Template
import pycuda.autoinit
import pycuda.gpuarray

from petri_net import SPN


class TlArgs:
    def __init__(self):
        # definitions of these variables are in cuTauLeaping paper by Nobile
        # et al.
        self.c = []
        self.x_0 = []

        self.A = []  # flattened spn.Pre matrix
        self.V = []
        self.V_t = []
        self.V_bar = []

        self.M = 0
        self.N = 0
        self.A_size = 0
        self.V_size = 0
        self.V_t_size = 0
        self.V_bar_size = 0

        self.hazards = []

        self.H = []
        self.H_type = []

        # these should be taken from a simulation set up xml
        self.ita = 100  # number of time recording points
        self.kappa = 1  # number of species we're recording
        self.n_c = 10  # tauLeaping critical reaction threshold, default=10
        self.eta = 0.03  # tauLeaping error control param, default=0.03
        self.t_max = 10  # the time at which the simulation ends

        self.E = [1]  # indices of the output species

        self.U = 500


class TlParser:
    @staticmethod
    def parse(sbml_model):
        """

        :param sbml_model:
        :return: args_out:
        """

        # THESE ARE TAKEN FROM THE SBML_MODEL
        stochastic_petri_net = SPN()
        stochastic_petri_net.sbml_2_stochastic_petri_net(sbml_model)

        args_out = TlArgs()

        args_out.c = numpy.array(stochastic_petri_net.c).astype(numpy.float32)

        args_out.x_0 = numpy.array(stochastic_petri_net.M).astype(numpy.uint32)

        ma = stochastic_petri_net.Pre
        mb = stochastic_petri_net.Post
        mv = mb - ma
        mv_t = mv.transpose()
        mv_bar = abs(mv * stochastic_petri_net.StoichMConst)

        args_out.A, args_out.A_size = TlParser.flatten_matrix(ma)
        args_out.V, args_out.V_size = TlParser.flatten_matrix(mv)
        args_out.V_t, args_out.V_t_size = TlParser.flatten_matrix(mv_t)
        args_out.V_bar, args_out.V_bar_size = TlParser.flatten_matrix(mv_bar)

        args_out.M = len(stochastic_petri_net.T)
        args_out.N = len(stochastic_petri_net.P)

        args_out.H, args_out.H_type = TlParser.get_hors(stochastic_petri_net)

        args_out.hazards = TlParser.define_hazards(stochastic_petri_net)

        # THESE ARE TAKEN FROM THE SIMULATION XML
        # TODO
        # args_out.ita = 0
        # args_out.kappa = 0
        #
        # args_out.n_c = 0
        # args_out.eta = 0
        # args_out.t_max = 0
        #
        # args_out.E = []

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
                                              'species' + str(i - 1),
                                              'x[' + str(i - 1) + ']')
        for i in range(len(spn.c), 0, -1):
            hazards_mod_code = string.replace(hazards_mod_code,
                                              'param' + str(i - 1),
                                              'c[' + str(i - 1) + ']')

        # mymod = SourceModule(hazards_mod_code)
        #print hazards_mod_code
        return hazards_mod_code

    @staticmethod
    def get_reaction_orders(hazards, species):
        reaction_orders = []
        # parse each hazard equation to determine the order of the reaction
        for reaction_id, hazard in enumerate(hazards):
            # we need to 'pythonify' the power sign in the hazard string
            py_hazard = string.replace(hazard, '^', '**')

            # expand the equation to it's canonical form
            py_hazard = str(sympy.expand(py_hazard))
            # is degree of poly == reaction order?
            temp_poly = sympy.poly(py_hazard, sympy.symbols(species.keys()))
            poly_degree = sympy.polys.polytools.Poly.total_degree(temp_poly)
            reaction_orders.append(poly_degree)
        return reaction_orders

    @staticmethod
    def get_hors(spn):
        hors = []
        hors_type = []

        reaction_orders = TlParser.get_reaction_orders(spn.h, spn.P)
        # for each reaction, check if a species is in it
        for reaction, reaction_idx in spn.T.iteritems():
            for species, species_idx in spn.P.iteritems():
                hazard = spn.h[reaction_idx]
                # check if the species is in the reaction
                match = re.search('(?<![a-zA-Z0-9_])' + species +
                                  '(?![a-zA-Z0-9_])', hazard)
                if match is not None:
                    # check if the current reaction order associated with a
                    # species is less than that of this reaction
                    if hors[species_idx] <= reaction_orders[reaction_idx]:
                        hors[species_idx] = reaction_orders[reaction_idx]

                    # get the stoichiometry of this species in this reaction
                    stoich = spn.Pre[species_idx][reaction_idx]
                    if stoich > hors_type[species_idx]:
                        hors_type[species_idx] = int(stoich)

        return numpy.array(hors), numpy.array(hors_type)

    @staticmethod
    def flatten_matrix(matrix):
        # each entry in the flattened matrix is an uchar4
        f_matrix = []

        count = 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i][j] != 0:
                    entry = pycuda.gpuarray.vec.make_char4(int(i), int(j),
                                                           int(matrix[i][j]), 0)
                    f_matrix.append(entry)
                    count += 1
        return numpy.array(f_matrix), count

##########
# TEST
##########
# import libsbml
#
# # sbml_file = '/home/sandy/Downloads/BIOMD0000000001_SBML-L3V1.xml'
# sbml_file = '/home/sandy/Documents/Code/cuda-sim-code/examples/ex02_p53
# /p53model.xml'
# reader = libsbml.SBMLReader()
# document = reader.readSBML(sbml_file)
# # check the SBML for errors
# error_count = document.getNumErrors()
# if error_count > 0:
# raise UserWarning(error_count + ' errors in SBML file: ' + open_file_.name)
# sbml_model = document.getModel()
#
# my_args = TlParser.parse(sbml_model)
# spn = SPN()
# spn.sbml_2_stochastic_petri_net(sbml_model)
# # print spn.Pre
# print my_args.__dict__
# print spn.h
# print spn.P
