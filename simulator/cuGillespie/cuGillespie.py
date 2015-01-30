import string
import numpy

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

import gillespie
from parser import GParser
from simulator import StochasticSimulator


class CuGillespie(StochasticSimulator):
    def run(self, sbml_model):
        my_args = GParser.parse(sbml_model)

        grid_size, block_size = self.CalculateSizes(my_args)

        gillespie_template = string.Template(gillespie.kernel)
        gillespie_code = gillespie_template.substitute(V_SIZE=my_args.V_size,
                                                       SPECIES_NUM=my_args.N,
                                                       THREAD_NUM=my_args.U,
                                                       PARAM_NUM=len(my_args.c),
                                                       REACT_NUM=my_args.M,
                                                       KAPPA=my_args.kappa,
                                                       ITA=my_args.ita,
                                                       BLOCK_SIZE=block_size,
                                                       T_MAX=my_args.t_max,
                                                       UPDATE_PROPENSITIES=my_args.hazards)

        # print gillespie_code
        gillespie_kernel = SourceModule(gillespie_code, no_extern_c=True)

        self.LoadDataOnGPU(my_args, gillespie_kernel)

        d_O, d_t, d_F = self.AllocateDataOnGPU(my_args)
        d_rng = self.get_rng_states(my_args.U)

        kernel_Gillespie = gillespie_kernel.get_function('kernel_Gillespie')
        kernel_Gillespie(d_O, d_t, d_F, d_rng, grid=(int(grid_size), 1, 1),
                         block=(int(block_size), 1, 1))

        O = d_O.get()
        return O


    def LoadDataOnGPU(self, tl_args, module):
        d_V = module.get_global('d_V')[0]
        cuda.memcpy_htod(d_V, tl_args.V)

        d_c = module.get_global('d_c')[0]
        cuda.memcpy_htod(d_c, tl_args.c)

        d_I = module.get_global('d_I')[0]
        cuda.memcpy_htod(d_I, tl_args.I)

        d_E = module.get_global('d_E')[0]
        cuda.memcpy_htod(d_E, tl_args.E)

        d_x_0 = module.get_global('d_x_0')[0]
        cuda.memcpy_htod(d_x_0, tl_args.x_0)


    def AllocateGlobals(self, O, t, F):
        d_O = gpuarray.to_gpu(O)
        d_t = gpuarray.to_gpu(t)
        d_F = gpuarray.to_gpu(F)

        return d_O, d_t, d_F


    def AllocateDataOnGPU(self, tl_args):
        O = numpy.zeros([tl_args.kappa, tl_args.ita, tl_args.U], numpy.uint32)
        t = numpy.zeros(tl_args.U, numpy.float32)
        F = numpy.zeros(tl_args.U, numpy.uint32)

        return self.AllocateGlobals(O, t, F)

##########
# TEST
##########
# import libsbml
#
# sbml_file = '/home/sandy/Downloads/simple_sbml.xml'
# reader = libsbml.SBMLReader()
# document = reader.readSBML(sbml_file)
# # check the SBML for errors
# error_count = document.getNumErrors()
# if error_count > 0:
#     raise UserWarning(error_count + ' errors in SBML file: ' + open_file_.name)
# sbml_model = document.getModel()
#
# O = cu_gillespie(sbml_model)
# # print O
#
# import matplotlib.pyplot as plt
#
# num_bins = 70
# # the histogram of the data
# n, bins, patches = plt.hist(O[2][1], num_bins, normed=1, facecolor='green',
#                             alpha=0.5)
# plt.axis([10, 90, 0, 0.07])
# plt.subplots_adjust(left=0.15)
# plt.show()