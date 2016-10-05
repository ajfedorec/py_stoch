import numpy as np
import libsbml


class SPN:
    def __init__(self):
        self.Pre = []
        self.Post = []
        self.M = []
        self.h = []
        self.c = []
        self.P = {}
        self.T = {}

        self.StoichMConst = []

    def sbml_2_stochastic_petri_net(self, sbml_model):

        # get reaction reactant stoichiometry matrix Pre
        # get reaction product stoichiometry matrix Post
        self.Pre, self.Post, self.StoichMConst = self.get_stoichiometries(
            sbml_model)

        # get initial network state vector M
        self.M = self.get_initial_state_vector(sbml_model)
        # print self.M

        # get rate constants vector c
        self.c = self.get_rate_constants_vector(sbml_model)
        constants = self.get_rate_constants(sbml_model)
        # print self.c

        # get species vector P
        self.P = self.get_species_vector(sbml_model)
        # self.P = self.M.keys()
        # print self.P

        # get reactions vector T
        self.T = self.get_reactions_vector(sbml_model)
        # self.T = self.h.keys()
        # print self.T

        # get hazards vector h
        self.h = self.get_hazards_vector(sbml_model, self.P, constants)
        # print self.h

    @staticmethod
    def get_species_vector(sbml_model):
        p = {}
        for species_idx, species in enumerate(sbml_model.getListOfSpecies()):
            p["species_" + str(species_idx)] = species_idx

        return p

    @staticmethod
    def get_reactions_vector(sbml_model):
        t = {}
        for reaction_idx, reaction in enumerate(
                sbml_model.getListOfReactions()):
            t[reaction.getId()] = reaction_idx

        return t

    @staticmethod
    def get_rate_constants(sbml_model):
        con = {}
        for param_idx, param in enumerate(sbml_model.getListOfParameters()):
            con[param.getId()] = param_idx
        return con

    @staticmethod
    def get_stoichiometries(sbml_model):
        reactants_matrix = np.zeros([len(sbml_model.getListOfSpecies()),
                                     len(sbml_model.getListOfReactions())])
        products_matrix = np.zeros_like(reactants_matrix)
        constant_matrix = np.zeros_like(reactants_matrix)

        for reaction_idx, reaction in enumerate(
                sbml_model.getListOfReactions()):
            reactants = {r.getSpecies(): r.getStoichiometry() for r in
                         reaction.getListOfReactants()}
            products = {p.getSpecies(): p.getStoichiometry() for p in
                        reaction.getListOfProducts()}

            for species_idx, species in enumerate(
                    sbml_model.getListOfSpecies()):
                species_id = species.getId()
                if species_id in reactants.keys():
                    reactants_matrix[species_idx, reaction_idx] = \
                        int(reactants.get(species_id, 0))
                if species_id in products.keys():
                    products_matrix[species_idx, reaction_idx] = \
                        int(products.get(species_id, 0))
                if species.getConstant():
                    constant_matrix[species_idx, reaction_idx] = 1

        return reactants_matrix, products_matrix, constant_matrix

    @staticmethod
    def get_initial_state_vector(sbml_model):
        m = []
        for species_idx, species in enumerate(sbml_model.getListOfSpecies()):
            m.append(species.getInitialAmount())
        return m

    @staticmethod
    def get_hazards_vector(sbml_model, species_dict, param_dict):
        h = []
        for reaction_idx, reaction in enumerate(
                sbml_model.getListOfReactions()):
            math = SPN.replace_hazard_species(
                reaction.getKineticLaw().getMath(),
                sbml_model)
            math = SPN.replace_hazard_parameter(math, param_dict)
            math = libsbml.formulaToL3String(math)
            h.append(math)
        return h

    @staticmethod
    def replace_hazard_species(mathml, sbml_model):
        num_children = mathml.getNumChildren()
        species_list = sbml_model.getListOfSpecies()
        for i in range(num_children):
            SPN.replace_hazard_species(mathml.getChild(i), sbml_model)
        if mathml.getType() == libsbml.AST_NAME:
            for species_idx, species in enumerate(species_list):
                if species.getId() == mathml.getName():
                    mathml.setName('species_' + str(species_idx))
                    break
        return mathml

    @staticmethod
    def replace_hazard_parameter(mathml, params_dict):
        num_children = mathml.getNumChildren()
        for i in range(num_children):
            SPN.replace_hazard_parameter(mathml.getChild(i), params_dict)
        if mathml.getType() == libsbml.AST_NAME:
            for param, param_idx in params_dict.iteritems():
                if param == mathml.getName():
                    mathml.setName('parameter_' + str(param_idx))
                    break
        return mathml

    @staticmethod
    def get_rate_constants_vector(sbml_model):
        c = []
        for constant_idx, constant in enumerate(
                sbml_model.getListOfParameters()):
            c.append(constant.getValue())
        return c


##########
# TEST
##########
# sbml_file = '/home/sandy/Downloads/BIOMD0000000001_SBML-L3V1.xml'
# reader = libsbml.SBMLReader()
# document = reader.readSBML(sbml_file)
# # check the SBML for errors
# error_count = document.getNumErrors()
# if error_count > 0:
# raise UserWarning(error_count + ' errors in SBML file: ' +
# open_file_.name)
# sbml_model = document.getModel()
#
# SPN().sbml_2_stochastic_petri_net(sbml_model)

