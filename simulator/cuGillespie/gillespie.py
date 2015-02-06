kernel = '''
#include <stdio.h>
#include <curand_kernel.h>

extern "C" {
// CONSTANT MEMORY
__device__ __constant__ float d_c[$PARAM_NUM];
__device__ __constant__ uint d_x_0[$SPECIES_NUM];
__device__ __constant__ float d_I[$ITA + 1];  // +1 because I include 0 timepoint
__device__ __constant__ unsigned char d_E[$KAPPA];
__device__ __constant__ char4 d_V[$V_SIZE];

// FUNCTION DECLARATIONS
__device__ void UpdatePropensities(float* a, uint* x, float* c);
__device__ uint SingleCriticalReaction(float* a, float a_0,
curandStateMRG32k3a* rstate);
__device__ void SaveDynamics(uint* x, uint f, uint tid, uint O[$KAPPA][$ITA][$THREAD_NUM]);
__device__ void UpdateState(uint* x, uint j);

__global__ void kernel_Gillespie(uint d_O[$KAPPA][$ITA][$THREAD_NUM],
                                 float d_t[$THREAD_NUM],
                                 uint d_F[$THREAD_NUM],
                                 curandStateMRG32k3a d_rng[$THREAD_NUM])
{
    // tid <- getGloablId()
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    // sid <- getLocalId()
    int sid = threadIdx.x;

    __shared__ curandStateMRG32k3a rstate[$BLOCK_SIZE];
    rstate[sid] = d_rng[tid];

    // x[sid] <- x_0
    __shared__ uint x[$BLOCK_SIZE][$SPECIES_NUM];
    for(int i = 0; i < $SPECIES_NUM; i++)
    {
        x[sid][i] = d_x_0[i];
    }

    // c[sid] <- c
    __shared__ float c[$BLOCK_SIZE][$PARAM_NUM];
    for(int i = 0; i < $PARAM_NUM; i++)
    {
        c[sid][i] = d_c[i];
    }

    // while t < T_max
    while(d_t[tid] < $T_MAX)
    {
        // calculate h_i(x_i, c_i)
        __shared__ float a[$BLOCK_SIZE][$REACT_NUM];
        for(int i = 0; i < $REACT_NUM; i++)
        {
            a[sid][i] = 0.;
        }
        UpdatePropensities(a[sid], x[sid], c[sid]);

        // h_0(x, c) = sum(h_i(x_i, c_i)
        float a_0 = 0.;
        for(int i = 0; i < $REACT_NUM; i++)
        {
            a_0 = a_0 + a[sid][i];
        }

        // 18. Tau <- (1./a_0) * ln(1./rho_1)
        float tau = 1000000.;
        float rho_1 = curand_uniform(&rstate[sid]);
        tau = (1. / a_0) * logf(1. / rho_1);
        //printf("tau = %f\\n", tau);


        // 20. T[tid] <- T[tid] + Tau
        d_t[tid] = d_t[tid] + tau;
        //printf("t = %f\\n", d_t[tid]);

        // 19. j <- SingleCriticalReaction( xeta[sid], a[sid] )
        uint j = SingleCriticalReaction(a[sid], a_0, &rstate[sid]);

        // 21. if T[tid] >= I[F[tid]] then
        if(d_t[tid] >= d_I[d_F[tid]])
        {
            //printf("d_t = %f,  d_I[d_F] = %f, d_F = %d in thread %d\\n", d_t[tid], d_I[d_F[tid]], d_F[tid], tid);
            // 22. SaveDynamics(x[sid],O[tid],E[tid])
            SaveDynamics(x[sid], d_F[tid], tid, d_O);

            //23. F[tid]++
            d_F[tid] = d_F[tid] + 1;
            if(d_F[tid] == $ITA)
            {
                return;
            }
        // 27. end if
        }

        // 28. x <- UpdateState( x, j )
        UpdateState(x[sid], j);
    }
}

__device__ uint SingleCriticalReaction(float* a, float a_0,
                                       curandStateMRG32k3a* rstate)
{
    float rho = curand_uniform(rstate);
    float rcount = 0;

    for(int j = 0; j < $REACT_NUM; j++)
    {
        rcount = rcount + a[j] / a_0;
        if(rcount >= rho)
        {
            return j;
        }
    }
}

__device__ void SaveDynamics(uint* x, uint f, uint tid,
                             uint O[$KAPPA][$ITA][$THREAD_NUM])
{
    //printf("save dynamics f = %d in thread %d\\n", f, tid);
    for(int i = 0; i < $KAPPA; i++)
    {
        //printf("i: %d, d_E[%d]: %d, x[%d]: %d, f: %d\\n", i, i, d_E[i], d_E[i], x[d_E[i]], f);
        O[i][f][tid] = x[d_E[i]];
    }
}

__device__ void UpdateState(uint* x, uint j)
{
    // we loop over each non-zero element in the stoichiometry matrix
    for(int v = 0; v < $V_SIZE; v++)
    {
        if(d_V[v].y == j)
        {
            x[d_V[v].x] = x[d_V[v].x] + d_V[v].z;
        }
    }
}

$UPDATE_PROPENSITIES
}
'''
