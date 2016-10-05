from mod.parser import TlParser, TlArgs
from mod.petri_net import SPN
import numpy


class TLArgsList:
    @staticmethod
    def make_list(sbml_model, sweep_species_init, sweep_params, output_species,
                  duration, num_output_time_points, num_simulations,
                  gillespie=0):
        tl_args_list = []

        species_init_lists = rest_list(sweep_species_init)
        param_lists = rest_list(sweep_params)

        sim_objects = []
        for species_init_list in species_init_lists:
            for param_list in param_lists:
                sim_object = SimObj()
                sim_object.species_init = species_init_list
                sim_object.params = param_list

                sim_objects.append(sim_object)

        for sim_object in sim_objects:
            spn = SPN.SPN()
            spn.sbml_2_stochastic_petri_net(sbml_model)
            spn.M = numpy.array(sim_object.species_init).astype(numpy.int32)
            spn.c = numpy.array(sim_object.params).astype(numpy.float64)

            tl_args = TlArgs()
            tl_args.E = numpy.array(output_species).astype(numpy.int32)
            tl_args.t_max = duration
            tl_args.ita = num_output_time_points
            tl_args.U = num_simulations

            tl_args.gillespie = gillespie

            tl_args = TlParser.parse(tl_args, spn)

            tl_args_list.append(tl_args)

        return tl_args_list


def rest_list(list_in, index=0):
    if index == len(list_in) - 1:
        return list_in[index]
    sub_lists = rest_list(list_in, index+1)
    my_list = []
    for i in range(len(list_in[index])):
        for j in range(len(sub_lists)):
        #     new_sublist  = []
        #     if isinstance(sub_lists[j], list):
        #         new_sublist = [list_in[index][i]] + sub_lists[j]
        #     else:
        #         new_sublist = [list_in[index][i]] + [sub_lists[j]]
        #     my_list.append(new_sublist)

            # this is a bit more Pythonic
            my_list.append([list_in[index][i]] + sub_lists[j]
                           if isinstance(sub_lists[j], list)
                           else [list_in[index][i]] + [sub_lists[j]])
    return my_list


class SimObj:
    def __init__(self):
        self.species_init = []
        self.params = []
