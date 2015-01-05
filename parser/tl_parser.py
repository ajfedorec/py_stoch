# VARIABLE | DESCRIPTION                       | TYPE  | DIMENSIONS | KERNEL
#
# ********** MODEL VARIABLES **********
# x        - system state                      - uint  - N          - P1-P2, P3 //
# a        - values of propensity functions    - float - M          - P1-P2, P3 // COALESCED
# x_prime  - putative system state             - int   - N          - P1-P2     // SHARED
# xeta     - critical reactions                - {0,1} - M          - P1-P2     // MEMORY
# K        - Poisson samples for reactions     - uint  - M          - P1-P2     //
#
# t        - simulation time of each thread    - float - U            // MAYBE GLOBAL?
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
# G        - used to calculate tau             - float - N            // GLOBAL MEM
#
# ********** PARAMETERS OF THE SIMULATION **********
# n_c      - critical reaction threshold       - uint  - 1            //  CONSTANT
# eta      - error control parameter           - uint  - 1            //  MEMORY
# t_max    - simulation length                 - float?- 1            //
#
# I    - time instants for output              - float - ita              //
# E    - indices of the output species         - uint  - kappa            //  GLOBAL
# O    - outputs                               - uint  - kappa x ita x U  //  MEM
# F    - pointer to next time instant          - uint  - U                //
# Q    -                                       - {-1,0,1} - U             //

from petri_net import SPN


class TlArgs:
    def __init__(self):
        self.c = []
        self.x_0 = []

        self.A = []  # flattened Pre matrix
        self.V = []
        self.V_t = []
        self.V_bar = []

        self.ita = 0
        self.kappa = 0
        self.M = 0
        self.N = 0
        self.A_size = 0
        self.V_size = 0
        self.V_t_size = 0
        self.V_bar_size = 0

        self.H = []
        self.H_type = []

        self.G = []

        self.n_c = 0
        self.eta = 0
        self.t_max = 0

        self.I = []
        self.E = []


class TlParser:
    @staticmethod
    def parse(sbml_model):
        # THESE ARE TAKEN FROM THE SBML_MODEL
        stochastic_petri_net = SPN()
        stochastic_petri_net.sbml_2_stochastic_petri_net(sbml_model)

        args_out = TlArgs()

        args_out.c = stochastic_petri_net.c
        args_out.x_0 = stochastic_petri_net.M

        ma = stochastic_petri_net.Pre
        mb = stochastic_petri_net.Post
        mv = mb - ma
        mv_t = mv.transpose()
        mv_bar = mv * stochastic_petri_net.StoichMConst

        args_out.A, args_out.A_size = TlParser.flatten_matrix(ma)
        args_out.V, args_out.V_size = TlParser.flatten_matrix(mv)
        args_out.V_t, args_out.V_t_size = TlParser.flatten_matrix(mv_t)
        args_out.V_bar, args_out.V_bar_size = TlParser.flatten_matrix(mv_bar)

        args_out.M = len(stochastic_petri_net.T)
        args_out.N = len(stochastic_petri_net.P)

        args_out.H, args_out.H_type = TlParser.get_hors(stochastic_petri_net)

        # THESE ARE TAKEN FROM THE SIMULATION XML
        # TODO
        args_out.ita = 0
        args_out.kappa = 0

        args_out.n_c = 0
        args_out.eta = 0
        args_out.t_max = 0

        args_out.I = []
        args_out.E = []

        return args_out

    @staticmethod
    def get_hors(spn):
        return hors, hors_type

    @staticmethod
    def flatten_matrix(matrix):
        return f_matrix

##########
# TEST
##########
# import libsbml
# sbml_file = '/home/sandy/Downloads/BIOMD0000000001_SBML-L3V1.xml'
# reader = libsbml.SBMLReader()
# document = reader.readSBML(sbml_file)
# # check the SBML for errors
# error_count = document.getNumErrors()
# if error_count > 0:
# raise UserWarning(error_count + ' errors in SBML file: ' + open_file_.name)
# sbml_model = document.getModel()
#
# TlParser.parse(sbml_model)
