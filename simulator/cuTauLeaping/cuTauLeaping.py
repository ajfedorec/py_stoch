import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from parser import TlParser


def cuTauLeaping(tl_args):
    return


cuTauLeaping_memoryDeclaration = '''
// CONSTANT MEMORY
__device__ __constant__ char4 d_A[$A_SIZE];
__device__ __constant__ char4 d_V[$V_SIZE];
__device__ __constant__ char4 d_V_t[$V_T_SIZE];
__device__ __constant__ char4 d_V_bar[$V_BAR_SIZE];
__device__ __constant__ uint d_H[$SPECIES_NUM]; // ?
__device__ __constant__ uint d_H_type[$SPECIES_NUM]; // ?

__device__ __constant__ uint d_ita;
__device__ __constant__ uint d_kappa;
__device__ __constant__ uint d_M;
__device__ __constant__ uint d_N;
__device__ __constant__ uint d_n_c;
__device__ __constant__ uint d_eta;
__device__ __constant__ uint d_A_size;
__device__ __constant__ uint d_V_size;
__device__ __constant__ uint d_V_t_size;
__device__ __constant__ uint d_V_bar_size;
__device__ __constant__ float d_t_max;

// GLOBAL MEMORY
__device__ float d_t[$THREAD_NUM];
__device__ uint d_x[$THREAD_NUM][$SPECIES_NUM];
__device__ uint d_O[$KAPPA][$ITA][$THREAD_NUM];
__device__ uint d_E[$KAPPA];
__device__ int d_Q[$THREAD_NUM];
'''

import string


def LoadDataOnGPU(tl_args):
    template = string.Template(cuTauLeaping_memoryDeclaration)
    code = template.substitute(A_SIZE=tl_args.A_size,
                               V_SIZE=tl_args.V_size,
                               V_T_SIZE=tl_args.V_t_size,
                               V_BAR_SIZE=tl_args.V_bar_size,
                               SPECIES_NUM=tl_args.N,
                               THREAD_NUM=tl_args.U,
                               KAPPA=tl_args.kappa,
                               ITA=tl_args.ita)
    module = SourceModule(code)

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


kernel = '''
__global__ void P1_P2(char4* A, char4* V, char4* V_t, char4* V_bar,
                      uint** global_x, float** global_c, float* I, uint* H,
                      uint* H_type, uint* E, uint*** O, char* Q, float* t)
{
    // tid <- getGlobalId()
    // I think this means an index of the thread on a global level rather than a
    // block level. Could be:
    //    threadIdx.x + (blockIdx.x * blockDim.x)
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    // sid <- getLocalId()
    // I think this means the index of the thread within the block
    int sid = threadIdx.x;

    // x[sid] <- global_x[tid]
    uint[d_N]* x;
    uint* x[sid] = global_x[tid];

    // c[sid] <- global_c[tid]
    //float* c[sid] = global_c[tid];

    // if Q[tid] = -1 then return
    //if Q[tid] == -1
    //{
    //    return;
    //}

    //printf("%d", tid);
    return;
}
'''

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

my_args = TlParser.parse(sbml_model)

LoadDataOnGPU(my_args)

kernel_code = SourceModule(kernel)
mykernel = kernel_code.get_function("P1_P2")