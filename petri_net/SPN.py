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
        self.Pre, self.Post, self.StoichMConst = self.get_stoichiometries(sbml_model)
        # print self.Pre
        # print self.Post

        # get initial network state vector M
        self.M = self.get_initial_state_vector(sbml_model)
        # print self.M

        # get hazards vector h
        self.h = self.get_hazards_vector(sbml_model)
        # print self.h

        # get rate constants vector c
        self.c = self.get_rate_constants_vector(sbml_model)
        # print self.c

        # get species vector P
        self.P = self.get_species_vector(sbml_model)
        # self.P = self.M.keys()
        # print self.P

        # get reactions vector T
        self.T = self.get_reactions_vector(sbml_model)
        # self.T = self.h.keys()
        # print self.T

    def get_species_vector(self, sbml_model):
        p = {}
        for species_idx, species in enumerate(sbml_model.getListOfSpecies()):
            p[species.getId()] = species_idx

        return p

    def get_reactions_vector(self, sbml_model):
        t = {}
        for reaction_idx, reaction in enumerate(sbml_model.getListOfReactions()):
            t[reaction.getId()] = reaction_idx

        return t

    def get_stoichiometries(self, sbml_model):
        reactants_matrix = np.zeros([len(sbml_model.getListOfSpecies()),
                                     len(sbml_model.getListOfReactions())])
        products_matrix = np.zeros_like(reactants_matrix)
        constant_matrix = np.ones_like(reactants_matrix)

        for reaction_idx, reaction in enumerate(sbml_model.getListOfReactions()):
            reactants = {r.getSpecies(): r.getStoichiometry() for r in
                         reaction.getListOfReactants()}
            products = {p.getSpecies(): p.getStoichiometry() for p in
                        reaction.getListOfProducts()}

            for species_idx, species in enumerate(sbml_model.getListOfSpecies()):
                species_id = species.getId()
                if species_id in reactants.keys():
                    reactants_matrix[species_idx, reaction_idx] = int(reactants.get(species_id, 0))
                if species_id in products.keys():
                    products_matrix[species_idx, reaction_idx] = int(products.get(species_id, 0))
                if species.getConstant():
                    constant_matrix[species_idx, reaction_idx] = 0

        return reactants_matrix, products_matrix, constant_matrix

    def get_initial_state_vector(self, sbml_model):
        m = []
        for species_idx, species in enumerate(sbml_model.getListOfSpecies()):
            m.append(species.getInitialAmount())
        return m

    def get_hazards_vector(self, sbml_model):
        h = []
        for reaction_idx, reaction in enumerate(sbml_model.getListOfReactions()):
            h.append(libsbml.formulaToL3String(reaction.getKineticLaw().getMath()))
        return h

    def get_rate_constants_vector(self, sbml_model):
        c = []
        for constant_idx, constant in enumerate(sbml_model.getListOfParameters()):
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

