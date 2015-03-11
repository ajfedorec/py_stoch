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

from parser import Parser
from mod.petri_net import SPN


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
        self.E = numpy.array([0]).astype(
            numpy.int32)  # indices of the output species
        self.t_max = 1  # the time at which the simulation ends
        self.ita = 11  # number of time recording points
        self.U = 128

        # these are derived from setup params above
        self.kappa = len(self.E)  # number of species we're recording
        self.I = numpy.array([])  # instances for recording

        # these are default values that I won't change for now
        self.n_c = 10  # tauLeaping critical reaction threshold, default=10
        self.eta = 0.03  # tauLeaping error control param, default=0.03


class TlParser(Parser):
    @staticmethod
    def parse(sbml_model, settings_file):
        # THESE ARE TAKEN FROM THE SBML_MODEL
        stochastic_petri_net = SPN.SPN()
        stochastic_petri_net.sbml_2_stochastic_petri_net(sbml_model)

        args_out = TlArgs()

        args_out.c = numpy.array(stochastic_petri_net.c).astype(numpy.float64)

        args_out.x_0 = numpy.array(stochastic_petri_net.M).astype(numpy.int32)

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
        Parser.parse_settings(settings_file, args_out)

        args_out.kappa = len(args_out.E)  # number of species we're recording
        args_out.I = numpy.linspace(0, args_out.t_max, num=args_out.ita).astype(
            numpy.float32)
        return args_out


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
        hors = numpy.zeros(len(spn.P))
        hors_type = numpy.zeros(len(spn.P))
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
        return numpy.array(hors).astype(numpy.int32), numpy.array(
            hors_type).astype(numpy.int32)

##########
# TEST
##########
# import libsbml
#
# sbml_file = '/home/sandy/Downloads/plasmid_stability.xml'
# # sbml_file = '/home/sandy/Documents/Code/cuda-sim-code/examples/ex02_p53
# # /p53model.xml'
# reader = libsbml.SBMLReader()
# document = reader.readSBML(sbml_file)
# # check the SBML for errors
# error_count = document.getNumErrors()
# if error_count > 0:
# raise UserWarning(error_count + ' errors in SBML file: ' + open_file_.name)
# sbml_model = document.getModel()
#
# # my_args = TlParser.parse(sbml_model)
# spn = SPN()
# spn.sbml_2_stochastic_petri_net(sbml_model)
# # print spn.Pre
# # print my_args.__dict__
# print spn.h
# print spn.P
