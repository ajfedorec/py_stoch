import string

import pycuda.driver as cuda
import pycuda.tools as cuda_tools
from pycuda.compiler import SourceModule

from parser import TlParser
import P1_P2


# def cu_tau_leaping(sbml_model, sim_info):
def cu_tau_leaping(sbml_model):
    # 2. A, V, V_t, V_bar, H, H_type <- CalculateDataStructures(MA, MB, x_0, c)
    my_args = TlParser.parse(sbml_model)

    P1_P2_template = string.Template(P1_P2.kernel)
    P1_P2_code = P1_P2_template.substitute(A_SIZE=my_args.A_size,
                                           V_SIZE=my_args.V_size,
                                           V_T_SIZE=my_args.V_t_size,
                                           V_BAR_SIZE=my_args.V_bar_size,
                                           SPECIES_NUM=my_args.N,
                                           THREAD_NUM=my_args.U,
                                           PARAM_NUM=len(my_args.c),
                                           REACT_NUM=my_args.M,
                                           KAPPA=my_args.kappa,
                                           ITA=my_args.ita,
                                           UPDATE_PROPENSITIES=my_args.hazards)
    print P1_P2_code
    kernel_code = SourceModule(P1_P2_code, no_extern_c=True)

    # 3. LoadDataOnGPU( A, V, V_t, V_bar, H, H_type, x_0, c )
    LoadDataOnGPU(my_args, kernel_code)

    # 4. AllocateDataOnGPU( t, x, O, E, Q )
    AllocateDataOnGPU(kernel_code)

    # 5. gridSize, blockSize <- DistributeWorkload(U)
    grid_size, block_size = CalculateSizes(my_args)

    # 6. repeat
    # while TerminSimulations != my_args.U:
    # 7. Kernel_p1-p2<<<gridSize, blockSize>>>
    # 8.    ( A, V, V_t, V_bar, H, H_type, x, c, I, E, O, Q, t )
    kernel_P1_P2 = kernel_code.get_function('kernel_P1_P2')
    kernel_P1_P2(grid=(int(grid_size), 1, 1), block=(int(block_size), 1, 1))

    # TODO
    # 9. Kernel_p3<<<gridSize, blockSize>>>
    # 10.   ( A, V, x, c, I, E, O, Q, t )

    # TODO
    # 11. TerminSimulations <- Kernel_p4 <<<gridSize,blockSize>>>(Q)

    # 12. unitl TerminSimulations = U


# 13. end procedure

def LoadDataOnGPU(tl_args, module):
    d_A = module.get_global('d_A')[0]
    cuda.memcpy_htod(d_A, tl_args.A)

    d_V = module.get_global('d_V')[0]
    cuda.memcpy_htod(d_V, tl_args.V)

    d_V_t = module.get_global('d_V_t')[0]
    cuda.memcpy_htod(d_V_t, tl_args.V_t)

    d_V_bar = module.get_global('d_V_bar')[0]
    cuda.memcpy_htod(d_V_bar, tl_args.V_bar)

    d_H = module.get_global('d_H')[0]
    cuda.memcpy_htod(d_H, tl_args.H)

    d_H_type = module.get_global('d_H_type')[0]
    cuda.memcpy_htod(d_H_type, tl_args.H_type)

    d_x_0 = module.get_global('d_x_0')[0]
    cuda.memcpy_htod(d_x_0, tl_args.x_0)

    d_c = module.get_global('d_c')[0]
    cuda.memcpy_htod(d_c, tl_args.c)

    d_ita = module.get_global('d_ita')[0]
    cuda.memcpy_htod(d_ita, bytes(tl_args.ita))

    d_kappa = module.get_global('d_kappa')[0]
    cuda.memcpy_htod(d_kappa, bytes(tl_args.kappa))

    d_M = module.get_global('d_M')[0]
    cuda.memcpy_htod(d_M, bytes(tl_args.M))

    d_N = module.get_global('d_N')[0]
    cuda.memcpy_htod(d_N, bytes(tl_args.N))

    d_n_c = module.get_global('d_n_c')[0]
    cuda.memcpy_htod(d_n_c, bytes(tl_args.n_c))

    d_eta = module.get_global('d_eta')[0]
    cuda.memcpy_htod(d_eta, bytes(tl_args.eta))

    d_A_size = module.get_global('d_A_size')[0]
    cuda.memcpy_htod(d_A_size, bytes(tl_args.A_size))

    d_V_size = module.get_global('d_V_size')[0]
    cuda.memcpy_htod(d_V_size, bytes(tl_args.V_size))

    d_V_t_size = module.get_global('d_V_t_size')[0]
    cuda.memcpy_htod(d_V_t_size, bytes(tl_args.V_t_size))

    d_V_bar_size = module.get_global('d_V_bar_size')[0]
    cuda.memcpy_htod(d_V_bar_size, bytes(tl_args.V_bar_size))

    d_t_max = module.get_global('d_t_max')[0]
    cuda.memcpy_htod(d_t_max, bytes(tl_args.t_max))


def AllocateDataOnGPU(module):
    d_t = module.get_global('d_t')[0]

    d_x = module.get_global('d_x')[0]

    d_O = module.get_global('d_O')[0]

    d_E = module.get_global('d_E')[0]

    d_Q = module.get_global('d_Q')[0]

    d_F = module.get_global('d_F')[0]


def CalculateSizes(tl_args):
    import math

    max_threads_per_block = cuda_tools.DeviceData().max_threads
    max_shared_mem = cuda_tools.DeviceData().shared_memory
    warp_size = cuda_tools.DeviceData().warp_size

    # T <= floor(MAX_shared / (13M + 8N)) from cuTauLeaping paper eq (5)
    threads_per_block = math.floor(
        max_shared_mem / (13 * tl_args.M + 8 * tl_args.N))

    # optimal T is a multiple of warp size
    optimal_threads_per_block = min(
        math.floor(threads_per_block / warp_size) * warp_size,
        max_threads_per_block)

    # grid size is equal to the number of blocks we need
    grid_size = math.ceil(tl_args.U / optimal_threads_per_block)

    return grid_size, min(optimal_threads_per_block, tl_args.U)

##########
# TEST
##########
import libsbml

# sbml_file = '/home/sandy/Downloads/BIOMD0000000001_SBML-L3V1.xml'
sbml_file = '/home/sandy/Documents/Code/cuda-sim-code/examples/ex02_p53' \
            '/p53model.xml'
reader = libsbml.SBMLReader()
document = reader.readSBML(sbml_file)
# check the SBML for errors
error_count = document.getNumErrors()
if error_count > 0:
    raise UserWarning(error_count + ' errors in SBML file: ' + open_file_.name)
sbml_model = document.getModel()

cu_tau_leaping(sbml_model)