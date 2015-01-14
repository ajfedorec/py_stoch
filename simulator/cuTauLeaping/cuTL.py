import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from parser import TlParser


cuTauLeaping = '''
#include <stdio.h>
#include <stdlib.h>

#include <curand.h>
#include <curand_kernel.h>

extern "C" {
// CONSTANT MEMORY
__device__ __constant__ char4 d_A[$A_SIZE];
__device__ __constant__ char4 d_V[$V_SIZE];
__device__ __constant__ char4 d_V_t[$V_T_SIZE];
__device__ __constant__ char4 d_V_bar[$V_BAR_SIZE];
__device__ __constant__ uint d_H[$SPECIES_NUM]; // ?
__device__ __constant__ uint d_H_type[$SPECIES_NUM]; // ?

__device__ __constant__ uint d_x_0[$SPECIES_NUM];
__device__ __constant__ float d_c[$PARAM_NUM];


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
__device__ uint d_F[$THREAD_NUM];

__device__ void UpdatePropensities(float* a, uint* x, float* c);
__device__ void GetSpecies(uint* temp, uint* x, uint* E);
__device__ void DetermineCriticalReactions(uint* xeta, uint* x);
__device__ void CalculateMuSigma(uint* x, float* a, uint* xeta, float* mu,
                                 float* sigma2);
__device__ float CalculateTau(uint* xeta, uint* x, float* mu, float* sigma2);
__device__ float CalculateG(uint* x, int i);

__global__ void kernel_P1_P2()
{
    // 2. tid <- getGlobalId()
    // I think this means an index of the thread on a global level rather than a
    // block level. Could be:
    //    threadIdx.x + (blockIdx.x * blockDim.x)
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    // 3. sid <- getLocalId()
    // I think this means the index of the thread within the block
    int sid = threadIdx.x;

    // 4. x[sid] <- global_x[tid]
    uint* x;
    if(d_t[tid] == 0)
    {
        x = d_x_0;
    }
    else
    {
        x = d_x[tid];
    }

    // 5. c[sid] <- global_c[tid]
    float* c = d_c;

    // 6., 7.
    if(d_Q[tid] == -1)
    {
        printf("Signal of terminated simulation in thread %d\\n", tid);
        return;
    }

    // 8. a <- UpdatePropensities( x[sid], c{tid] )
    float a[$REACT_NUM];
    UpdatePropensities(a, x, c);

    // 9. a0 <- Sum(a)
    float a_0 = 0;
    for(int i = 0; i < $REACT_NUM; i++)
    {
        a_0 = a_0 + a[i];
    }

    // 10. if a0 = 0 then
    if(a_0 == 0)
    {
        // 11. for i <- F[tid]...ita do
        for(int i = d_F[tid]; i < d_ita; i++)
        {
            // 12., 13. O[tid][i] <- GetSpecies(x[sid], E)
            GetSpecies(&d_O[0][tid][i], x, d_E);
        }
        // 14., 15., 16. Q[tid] <- -1
        d_Q[tid] = -1;
        printf("No reactions can be applied in thread %d\\n", tid);
        return;
    }

    // 17. Xeta <- DetermineCriticalReactions( A, x[sid] )
    uint Xeta[$REACT_NUM];
    DetermineCriticalReactions(Xeta, x);

    // 18. mu, sigma <- CalculateMuSigma( x[sid], H, H_type, a[sid], Xeta[sid] )
    float mu[$A_SIZE];
    float sigma2[$A_SIZE];
    CalculateMuSigma(x, a, Xeta, mu, sigma2);

    // 19. Tau_1 <- CalculateTau(mu, sigma)
    float tau_1 = CalculateTau(Xeta, x, mu, sigma2);

    // 20. if Tau_1 < 10./a_0 then
    if(tau_1 < 10.0 / a_0)
    {
        // 21., 22. Q[tid] <- 0
        d_Q[tid] = 0;
        printf("SSA steps are more efficient in thread %d\\n", tid);
        return;
    }
    // 23.
    else
    {
        // 24., 25. Q[tid] <- 1
        d_Q[tid] = 1;
        printf("tau-leaping will be performed in thread %d\\n", tid);
    }

    // 26. K <- [], a_0_c <- 0
    float K[$REACT_NUM];
    float a_0_c = 0;

    // 27. for all j in Xeta do
    for(int j = 0; j < d_M; j++)
    {
        if(Xeta[j] == 1)
        {
            // 28., 29. a_0_c <- a_0_c + a[j]  Sum of propensitites of
            critical reactions
            a_0_c = a_0_c + a[j];
        }
    }

    // 30., 31. if a_0_c > 0 then Tau_2 <- (1./a_0_c) * ln(1./rho_1)
    curandStateMRG32k3a rstate;
    curand_init(0, tid, 0, &rstate);
    float tau_2 = 1000000.;
    if(a_0_c > 0)
    {
        float rho_1 = curand_uniform(&rstate);
        tau_2 = (1. / a_0_c) * logf(1. / rho_1);
    }




}

__device__ void GetSpecies(uint* temp, uint* x, uint* E)
{
    for(int i = 0; i < d_kappa; i++)
    {
        temp[i] = x[E[i]];
    }
}

__device__ void DetermineCriticalReactions(uint* xeta, uint* x)
{
    // we loop over each reactant, d_A
    for(int i = 0; i < d_A_size; i++)
    {
        // if the stoichiometry of the reactant (d_A[i].z) * our critical
        // reaction threshold (d_n_c) is greater than the current number of
        // molecules of the reactant species (x[d_A[i].x]), the reaction in
        // which the reactant is taking place is classed as critical.
        if((d_A[i].z * d_n_c) > x[d_A[i].x])
        {
            xeta[d_A[i].y] = 1;
        }
    }
}

__device__ void CalculateMuSigma(uint* x, float* a, uint* xeta, float* mu,
                                 float* sigma2)
{
    // we loop over each reactant
    for(int i = 0; i < d_A_size; i++)
    {
        // we loop over each non-zero element in the stoichiometry matrix
        for(int v = 0; v < d_V_size; v++)
        {
            // if the element in the stoichiometry matrix (d_V[v].x) is a
            // reactant (d_A[i].x) in the same reaction (d_V[v].y == d_A[i].y)
            if((d_V[v].x == d_A[i].x) && (d_V[v].y == d_A[i].y))
            {
                // if the reaction (d_V[v].y) is not critical
                if(xeta[d_V[v].y] == 0)
                {
                    // mu_i = Sum_j_ncr(v_ij * a_j(x)) for all i in {set of
                    reactant species}
                    // where: v_ij is the stoichiometry of species i in
                    reaction j
                    //   and a_j(x) is the propensity of reaction j given
                    state x
                    mu[d_V[v].x] = mu[d_V[v].x] + d_V[v].z * a[d_V[v].y];

                    // sigma^2_i = Sum_j_ncr(v_ij^2 * a_j(x)) for all i in {
                    set of reactant species}
                    sigma2[d_V[v].x] = mu[d_V[v].x] + d_V[v].z * d_V[v].z *
                    a[d_V[v].y];
                }
            }
        }
    }
}

__device__ float CalculateTau(uint* xeta, uint* x, float* mu, float* sigma2)
{
    // set initial tau to some large number, should be infinite
    float tau  = 1000000;

    // we loop over each reactant
    for(int i = 0; i < d_A_size; i++)
    {
        // if the reactant (d_A[i]) is in a non-critical reaction
        if(xeta[d_A[i].y] == 0)
        {
            float g_i = CalculateG(x, i);

            float numerator_l = fmaxf((d_eta * x[i] / g_i), 1);
            float lhs = numerator_l /  fabsf(mu[i]);
            float rhs = (numerator_l * numerator_l) / sigma2[i];
            float temp_tau = fminf(lhs, rhs);

            if(temp_tau < tau)
            {
                tau = temp_tau;
            }
        }
    }
    return tau;
}

__device__ float CalculateG(uint* x, int i)
{
    if(d_H[i] == 1)
    {
        return 1;
    }
    else if(d_H[i] == 2)
    {
        if(d_H_type[i] == 2)
        {
            return (2 + 1 / (x[i] - 1));
        }
        else
        {
            return 2;
        }
    }
    else if(d_H[i] == 3)
    {
        if(d_H_type[i] == 3)
        {
            return (3 + 1 / (x[i] - 1) + 2 / (x[i] - 2));
        }
        else if(d_H_type[i] == 2)
        {
            return 3/2 * (2 + 1 / (x[i] - 1));
        }
        else
        {
            return 3;
        }
    }
}
}
'''


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

    max_threads_per_block = cuda.Device(0).get_attribute(
        cuda.device_attribute.MAX_THREADS_PER_BLOCK)
    max_shared_mem = cuda.Device(0).get_attribute(
        cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
    warp_size = cuda.Device(0).get_attribute(cuda.device_attribute.WARP_SIZE)

    # T <= floor(MAX_shared / (13M + 8N)) from cuTauLeaping paper eq (5)
    threads_per_block = math.floor(
        max_shared_mem / (13 * tl_args.M + 8 * tl_args.N))

    # optimal T is a multiple of warp size
    optimal_threads_per_block = min(
        math.floor(threads_per_block / warp_size) * warp_size,
        max_threads_per_block)

    # grid size is equal to the number of blocks we need
    grid_size = math.ceil(tl_args.U / optimal_threads_per_block)

    return grid_size, optimal_threads_per_block


##########
# TEST
##########
import libsbml
import string

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

cuTauLeaping = cuTauLeaping + my_args.hazards
template = string.Template(cuTauLeaping)
code = template.substitute(A_SIZE=my_args.A_size,
                           V_SIZE=my_args.V_size,
                           V_T_SIZE=my_args.V_t_size,
                           V_BAR_SIZE=my_args.V_bar_size,
                           SPECIES_NUM=my_args.N,
                           THREAD_NUM=my_args.U,
                           PARAM_NUM=len(my_args.c),
                           REACT_NUM=my_args.M,
                           KAPPA=my_args.kappa,
                           ITA=my_args.ita)
print code
kernel_code = SourceModule(code, no_extern_c=True)
LoadDataOnGPU(my_args, kernel_code)
AllocateDataOnGPU(kernel_code)
grid_size, block_size = CalculateSizes(my_args)
# print grid_size, block_size

kernel_P1_P2 = kernel_code.get_function('kernel_P1_P2')
kernel_P1_P2(grid=(int(grid_size), 1, 1), block=(int(block_size), 1, 1))

#LoadDataOnGPU(my_args.A, my_args.V, my_args.V_t)