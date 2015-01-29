import string
import numpy

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.tools as cuda_tools
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

import gillespie
from parser import GParser


def cuGillespie(sbml_model):
    my_args = GParser.parse(sbml_model)

    gillespie_template = string.Template(gillespie.kernel)

    gillespie_code = gillespie_template.substitute(V_SIZE=my_args.V_size,
                                                   SPECIES_NUM=my_args.N,
                                                   THREAD_NUM=my_args.U,
                                                   PARAM_NUM=len(my_args.c),
                                                   REACT_NUM=my_args.M,
                                                   KAPPA=my_args.kappa,
                                                   ITA=my_args.ita,
                                                   BLOCK_SIZE=128,
                                                   T_MAX=my_args.t_max,
                                                   UPDATE_PROPENSITIES=my_args.hazards)

    # print gillespie_code
    gillespie_kernel = SourceModule(gillespie_code, no_extern_c=True)

    LoadDataOnGPU(my_args, gillespie_kernel)

    d_O, d_t, d_F = AllocateDataOnGPU(my_args)
    d_rng = get_rng_states(my_args.U)

    grid_size, block_size = CalculateSizes(my_args)

    kernel_Gillespie = gillespie_kernel.get_function('kernel_Gillespie')
    kernel_Gillespie(d_O, d_t, d_F, d_rng, grid=(int(grid_size), 1, 1),
                     block=(int(block_size), 1, 1))

    O = d_O.get()
    return O


def LoadDataOnGPU(tl_args, module):
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


def AllocateGlobals(O, t, F):
    d_O = gpuarray.to_gpu(O)
    d_t = gpuarray.to_gpu(t)
    d_F = gpuarray.to_gpu(F)

    return d_O, d_t, d_F


def AllocateDataOnGPU(tl_args):
    O = numpy.zeros([tl_args.kappa, tl_args.ita, tl_args.U], numpy.uint32)
    t = numpy.zeros(tl_args.U, numpy.float32)
    F = numpy.zeros(tl_args.U, numpy.uint32)

    return AllocateGlobals(O, t, F)


def CalculateSizes(tl_args):
    import math

    max_threads_per_block = cuda_tools.DeviceData().max_threads
    max_shared_mem = cuda_tools.DeviceData().shared_memory
    warp_size = cuda_tools.DeviceData().warp_size

    # T <= floor(MAX_shared / (13M + 8N)) from cuTauLeaping paper eq (5)
    threads_per_block = math.floor(
        max_shared_mem / (13 * tl_args.M + 8 * tl_args.N))

    # optimal T is a multiple of warp size
    max_optimal_threads_per_block = min(
        math.floor(threads_per_block / warp_size) * warp_size,
        max_threads_per_block)

    blocks = math.ceil(tl_args.U / 128)


    # grid size is equal to the number of blocks we need
    # grid_size = math.ceil(tl_args.U / optimal_threads_per_block)

    return blocks, 128


def get_rng_states(size):
    from pycuda import characterize

    init_rng_src = """
    #include <curand_kernel.h>

    extern "C"
    {

    __global__ void init_rng(int nthreads, curandStateMRG32k3a *s)
    {
        int tid = threadIdx.x + (blockIdx.x * blockDim.x);

        if (tid >= nthreads)
        {
            return;
        }

        curand_init(tid, 0, 0, &s[tid]);
    }

    } // extern "C"
    """

    rng_states = cuda.mem_alloc(
        size * characterize.sizeof('curandStateMRG32k3a',
                                   '#include <curand_kernel.h>'))

    module = pycuda.compiler.SourceModule(init_rng_src, no_extern_c=True)
    init_rng = module.get_function('init_rng')

    init_rng(numpy.int32(size), rng_states, numpy.uint64(0), block=(64, 1, 1),
             grid=(size // 64 + 1, 1))

    return rng_states

##########
# TEST
##########
import libsbml

sbml_file = '/home/sandy/Downloads/simple_sbml.xml'
reader = libsbml.SBMLReader()
document = reader.readSBML(sbml_file)
# check the SBML for errors
error_count = document.getNumErrors()
if error_count > 0:
    raise UserWarning(error_count + ' errors in SBML file: ' + open_file_.name)
sbml_model = document.getModel()

O = cuGillespie(sbml_model)
# print O

import matplotlib.pyplot as plt

num_bins = 70
# the histogram of the data
n, bins, patches = plt.hist(O[2][1], num_bins, normed=1, facecolor='green',
                            alpha=0.5)
plt.axis([10, 90, 0, 0.07])
plt.subplots_adjust(left=0.15)
plt.show()