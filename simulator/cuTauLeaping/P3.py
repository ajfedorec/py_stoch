kernel = '''
#include <stdio.h>
#include <curand_kernel.h>

extern "C" {
// CONSTANT MEMORY
__device__ __constant__ char4 d_A[$A_SIZE];
__device__ __constant__ char4 d_V[$V_SIZE];
__device__ __constant__ unsigned char d_H[$SPECIES_NUM]; // ?
__device__ __constant__ unsigned char d_H_type[$SPECIES_NUM]; // ?

__device__ __constant__ float d_c[$PARAM_NUM];

__device__ __constant__ float d_I[$ITA + 1];  // +1 because I include 0 time
point
__device__ __constant__ unsigned char d_E[$KAPPA];


// FUNCTION DECLARATIONS
__device__ void UpdatePropensities(float* a, uint* x, float* c);
__device__ uint SingleCriticalReaction(float* a, float a_0,
curandStateMRG32k3a* rstate);
__device__ void UpdateState(uint* x, uint j);
__device__ void SaveDynamics(uint* x, uint f, uint tid, uint O[$KAPPA][$ITA][
$THREAD_NUM]);


__global__ void kernel_P3(uint global_x[$THREAD_NUM][$SPECIES_NUM],
                          uint d_O[$KAPPA][$ITA][$THREAD_NUM],
                          short d_Q[$THREAD_NUM],
                          float d_t[$THREAD_NUM],
                          uint d_F[$THREAD_NUM],
                          curandStateMRG32k3a d_rng[$THREAD_NUM])
{
    // 2. tid <- getGloablId()
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    // 3. sid <- getLocalId()
    int sid = threadIdx.x;

    curandStateMRG32k3a rstate = d_rng[tid];

    // 4. if Q[tid] != 0 then return
    if(d_Q[tid] != 0)
    {
        //printf("SSA steps not needed in thread %d\\n", tid);
        return;
    // 5. end if
    }

    // 6. x[sid] <- global_x[tid]
    __shared__ uint x[$BLOCK_SIZE][$SPECIES_NUM];
    for(int i = 0; i < $SPECIES_NUM; i++)
    {
        x[sid][i] = global_x[tid][i];
    }

    // 7. c[sid] <- global_c[tid]
    __shared__ float c[$BLOCK_SIZE][$PARAM_NUM];
    for(int i = 0; i < $PARAM_NUM; i++)
    {
        c[sid][i] = d_c[i];
    }

    // 8. for i in 0...100 do
    for(int counter = 0; counter < 100; counter++)
    {
        // 9. a <- UpdatePropensities( x[sid], c{tid] )
        __shared__ float a[$BLOCK_SIZE][$REACT_NUM];
        for(int i = 0; i < $REACT_NUM; i++)
        {
            a[sid][i] = 0.;
        }
        UpdatePropensities(a[sid], x[sid], c[sid]);

        // 10. a0 <- Sum(a)
        float a_0 = 0.;
        for(int i = 0; i < $REACT_NUM; i++)
        {
            a_0 = a_0 + a[sid][i];
        }

        // 11. if a_0 = 0 then
        if(a_0 == 0)
        {
            // 12. for i <- E[tid]...ita do
            for(int s = 0; s < $KAPPA; s++)
            {
                for(int f = d_F[tid]; f < $ITA; f++)
                {
                    // 13. O[tid][i] <- x[tid]
                    d_O[s][f][tid] = x[sid][d_E[s]];
                }
            // 14. end for
            }
            // 15. Q[tid] <- -1
            d_Q[tid] = -1;
            printf("SSA: No reactions can be applied in thread %d\\n", tid);
            // 16. return
            return;
        // 17. end if
        }

        // 18. Tau <- (1./a_0) * ln(1./rho_1)
        float tau = 1000000.;
        float rho_1 = curand_uniform(&rstate);
        tau = (1. / a_0) * logf(1. / rho_1);

        // 19. j <- SingleCriticalReaction( xeta[sid], a[sid] )
        uint j = SingleCriticalReaction(a[sid], a_0, &rstate);

        // 20. T[tid] <- T[tid] + Tau
        d_t[tid] = d_t[tid] + tau;

        // 21. if T[tid] >= I[F[tid]] then
        if(d_t[tid] >= d_I[d_F[tid]])
        {
            // 22. SaveDynamics(x[sid],O[tid],E[tid])
            SaveDynamics(x[sid], d_F[tid], tid, d_O);

            //23. F[tid]++
            d_F[tid] = d_F[tid] + 1;

            // 24. if F[tid] = ita then
            if(d_F[tid] == $ITA)
            {
                // 25. Q[tid] <- -1
                d_Q[tid] = -1;
                printf("SSA: No more samples: simulation over in thread
                %d\\n", tid);
            // 26. end if
            }
        // 27. end if
        }
        // 28. x <- UpdateState( x, j )
        UpdateState(x[sid], j);
    // 29. end for
    }
    // 30. global_x <- x
    for(int i = 0; i < $SPECIES_NUM; i++)
    {
        global_x[tid][i] = x[sid][i];
        //printf("global_x[%d][%d] = %d\\n", tid, i, global_x[tid][i]);
    }
// 31. end procedure
}

__device__ void SaveDynamics(uint* x, uint f, uint tid, uint O[$KAPPA][$ITA][
$THREAD_NUM])
{
    for(int i = 0; i < $KAPPA; i++)
    {
        O[i][f][tid] = x[d_E[i]];
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

