import string
import numpy
import math
from mod.parser import GParser

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.tools as tools

from mod.simulator import StochasticSimulator
from mod.simulator.cuGillespie import gillespie
from mod.utils import Timer


class CuGillespie(StochasticSimulator):
    def run(self, sbml_model, settings_file):
        my_args = GParser.parse(sbml_model, settings_file)

        grid_size, block_size = self.CalculateSizes(my_args)
        print grid_size, block_size

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

        gillespie_kernel = SourceModule(gillespie_code, no_extern_c=True)

        self.LoadDataOnGPU(my_args, gillespie_kernel)
        d_O = self.AllocateDataOnGPU(my_args)
        d_rng = self.get_rng_states(my_args.U)

        cuda.Context.synchronize()
        print cuda.mem_get_info()

        kernel_Gillespie = gillespie_kernel.get_function('kernel_Gillespie')
        kernel_Gillespie(d_O, d_rng, grid=(int(grid_size), 1, 1),
                     block=(int(block_size), 1, 1))
        cuda.Context.synchronize()


        # print cuda.mem_get_info()

        O = d_O.get()
        cuda.Context.synchronize()
        return O


    def LoadDataOnGPU(self, tl_args, module):
        d_V = module.get_global('d_V')[0]
        cuda.memcpy_htod_async(d_V, tl_args.V)

        d_c = module.get_global('d_c')[0]
        cuda.memcpy_htod_async(d_c, tl_args.c)

        d_I = module.get_global('d_I')[0]
        cuda.memcpy_htod_async(d_I, tl_args.I)

        d_E = module.get_global('d_E')[0]
        cuda.memcpy_htod_async(d_E, tl_args.E)

        d_x_0 = module.get_global('d_x_0')[0]
        cuda.memcpy_htod_async(d_x_0, tl_args.x_0)

    def AllocateGlobals(self, O):
        d_O = gpuarray.to_gpu_async(O)

        return d_O

    def AllocateDataOnGPU(self, tl_args):
        # d_O = gpuarray.zeros([tl_args.kappa, tl_args.ita, tl_args.U], numpy.uint32)
        # d_t = gpuarray.zeros(tl_args.U, numpy.float32)
        # d_F = gpuarray.zeros(tl_args.U, numpy.uint32)
        #
        # return d_O, d_t, d_F

        O = numpy.zeros([tl_args.kappa, tl_args.ita, tl_args.U], numpy.int32)

        return self.AllocateGlobals(O)

    def CalculateSizes(self, tl_args):
        hw_constrained_threads_per_block = tools.DeviceData().max_threads

        # T <= floor(MAX_shared / (13M + 8N)) from cuTauLeaping paper eq (5)
        # threads_per_block = math.floor(
        # max_shared_mem / (13 * tl_args.M + 8 * tl_args.N))
        # HOWEVER, for my implementation:
        #   type                size    var     number
        #   curandStateMRG32k3a 80      rstate  1
        #   uint                32      d_F     1
        #   double              64      d_t     1
        #   uint                32      x       N
        #   double              64      a       M
        #   T <= floor(Max_shared / (8M + 4N + 22) (bytes)
        max_shared_mem = tools.DeviceData().shared_memory
        shared_mem_constrained_threads_per_block = math.floor(max_shared_mem / (
            8 * tl_args.M + 4 * tl_args.N + 22))

        max_threads_per_block = min(hw_constrained_threads_per_block,
                                    shared_mem_constrained_threads_per_block)

        warp_size = tools.DeviceData().warp_size

        # optimal T is a multiple of warp size
        max_warps_per_block = math.floor(max_threads_per_block / warp_size)
        max_optimal_threads_per_block = max_warps_per_block * warp_size

        if (max_optimal_threads_per_block >= 256) and (tl_args.U >= 2560):
            block_size = 256
        elif max_optimal_threads_per_block >= 128 and (tl_args.U >= 1280):
            block_size = 128
        elif max_optimal_threads_per_block >= 64 and (tl_args.U >= 640):
            block_size = 64
        elif max_optimal_threads_per_block >= 32:
            block_size = 32
        else:
            block_size = max_optimal_threads_per_block

        if tl_args.U <= 2:
            block_size = tl_args.U

        grid_size = int(math.ceil(float(tl_args.U) / float(block_size)))

        tl_args.U = int(grid_size * block_size)

        return grid_size, block_size

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
