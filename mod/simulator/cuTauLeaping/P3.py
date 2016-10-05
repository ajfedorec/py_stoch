kernel = '''
#include <stdio.h>
#include <curand_kernel.h>

extern "C" {
// CONSTANT MEMORY
__device__ __constant__ char4 d_A[$A_SIZE]; // flattened Pre matrix
__device__ __constant__ char4 d_V[$V_SIZE]; // flattened Stoichiometry matrix
__device__ __constant__ int d_H[$SPECIES_NUM]; // HORs vector
__device__ __constant__ int d_H_type[$SPECIES_NUM]; //
__device__ __constant__ double d_c[$PARAM_NUM]; // parameters vector
__device__ __constant__ float d_I[$ITA];  // output time points
__device__ __constant__ int d_E[$KAPPA]; // output species indexes

// FUNCTION DECLARATIONS
__device__ void UpdatePropensities(double a[$REACT_NUM],
                                   int x[$SPECIES_NUM]);
__device__ int SingleCriticalReaction(double a[$REACT_NUM],
                                                double a_0,
                                                curandStateMRG32k3a* rstate);
__device__ void UpdateState(int x[$SPECIES_NUM],
                            int j);
__device__ void SaveDynamics(int x[$SPECIES_NUM],
                             int f,
                             int tid,
                             int O[$KAPPA][$ITA][$THREAD_NUM]);


__global__ void kernel_P3(int global_x[$THREAD_NUM][$SPECIES_NUM],
                          int d_O[$KAPPA][$ITA][$THREAD_NUM],
                          int d_Q[$THREAD_NUM],
                          double d_t[$THREAD_NUM],
                          int d_F[$THREAD_NUM],
                          curandStateMRG32k3a d_rng[$THREAD_NUM])
{
    // 2. tid <- getGlobalId()
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    // 3. sid <- getLocalId()
    int sid = threadIdx.x;

    __shared__ curandStateMRG32k3a rstate[$BLOCK_SIZE];
    rstate[sid] = d_rng[tid];

    // 4. if Q[tid] != 0 then return
    if(d_Q[tid] != 0)
    {
        //printf("SSA: d_Q = %d, SSA steps not needed in thread %d\\n",
        //        d_Q[tid], tid);
        return;
    // 5. end if
    }

    // 6. x[sid] <- global_x[tid]
    __shared__ int x[$BLOCK_SIZE][$SPECIES_NUM];
    #pragma unroll $SPECIES_NUM
    for(int species_idx = 0; species_idx < $SPECIES_NUM; species_idx++)
    {
        x[sid][species_idx] = global_x[tid][species_idx];
        //printf("SSA: x[%d] = %d at time %f in thread %d\\n", species_idx,
        //        x[sid][species_idx], d_t[tid], tid);
    }

    __shared__ double a[$BLOCK_SIZE][$REACT_NUM];
    double a_0 = 0.;
    double tau = 1000000.;
    double rho_1 = 0.;
    int j = 0;

    // 8. for i in 0...100 do
    for(int counter = 0; counter < 100; counter++)
    {
        // 9. a <- UpdatePropensities( x[sid], c{tid] )
        #pragma unroll $REACT_NUM
        for(int react_idx = 0; react_idx < $REACT_NUM; react_idx++)
        {
            a[sid][react_idx] = 0.;
        }
        UpdatePropensities(a[sid], x[sid]);

        // 10. a0 <- Sum(a)
        a_0 = 0.;
        #pragma unroll $REACT_NUM
        for(int react_idx = 0; react_idx < $REACT_NUM; react_idx++)
        {
            a_0 = a_0 + a[sid][react_idx];
        }

        // 11. if a_0 = 0 then
        if(a_0 == 0)
        {
            // 12. for i <- E[tid]...ita do
            #pragma unroll $KAPPA
            for(int species_out_idx = 0; species_out_idx < $KAPPA; species_out_idx++)
            {
                #pragma unroll $ITA
                for(int time_out_idx = d_F[tid]; time_out_idx < $ITA; time_out_idx++)
                {
                    // 13. O[tid][i] <- x[tid]
                    d_O[species_out_idx][time_out_idx][tid] = x[sid][d_E[species_out_idx]];
                }
            // 14. end for
            }
            // 15. Q[tid] <- -1
            d_Q[tid] = -1;
            //printf("SSA: No reactions can be applied in thread %d\\n", tid);
            // 16. return
            return;
        // 17. end if
        }

        // 18. Tau <- (1./a_0) * ln(1./rho_1)
        tau = 1e6;
        rho_1 = curand_uniform(&rstate[sid]);
        tau = (1. / a_0) * log(1. / rho_1);
        //printf("tau = %f\\n", tau);

        // 19. j <- SingleCriticalReaction( xeta[sid], a[sid] )
        j = SingleCriticalReaction(a[sid], a_0, &rstate[sid]);

        // 20. T[tid] <- T[tid] + Tau
        d_t[tid] = d_t[tid] + tau;
        //printf("t = %f\\n", d_t[tid]);

        // 21. if T[tid] >= I[F[tid]] then
        while(d_t[tid] >= d_I[d_F[tid]])
        {
            //printf("d_t = %f,  d_I[d_F] = %f, d_F = %d in thread %d\\n",
            //        d_t[tid], d_I[d_F[tid]], d_F[tid], tid);
            // 22. SaveDynamics(x[sid],O[tid],E[tid])
            SaveDynamics(x[sid], d_F[tid], tid, d_O);

            //23. F[tid]++
            d_F[tid] = d_F[tid] + 1;

            // 24. if F[tid] = ita then
            if(d_F[tid] == $ITA)
            {
                // 25. Q[tid] <- -1
                d_Q[tid] = -1;
                //printf("SSA: No more samples: simulation over in thread %d\\n", tid);
                return;
            // 26. end if
            }
        // 27. end if
        }

        // 28. x <- UpdateState( x, j )
        UpdateState(x[sid], j);
    // 29. end for
    }

    // 30. global_x <- x
    #pragma unroll $SPECIES_NUM
    for(int species_idx = 0; species_idx < $SPECIES_NUM; species_idx++)
    {
        global_x[tid][species_idx] = x[sid][species_idx];
        //printf("global_x[%d][%d] = %d\\n", tid, species_idx, global_x[tid][species_idx]);
    }

    d_rng[tid] = rstate[sid];
// 31. end procedure
}

__device__ void SaveDynamics(int x[$SPECIES_NUM], int f, int tid,
                             int O[$KAPPA][$ITA][$THREAD_NUM])
{
    //printf("save dynamics f = %d in thread %d\\n", f, tid);
    #pragma unroll $KAPPA
    for(int species_out_idx = 0; species_out_idx < $KAPPA; species_out_idx++)
    {
        //printf("i: %d, d_E[%d]: %d, x[%d]: %d, f: %d\\n", species_out_idx,
        //        species_out_idx, d_E[species_out_idx], d_E[species_out_idx],
        //        x[d_E[species_out_idx]], f);
        O[species_out_idx][f][tid] = x[d_E[species_out_idx]];
    }
}

__device__ int SingleCriticalReaction(double a[$REACT_NUM], double a_0,
                                      curandStateMRG32k3a* rstate)
{
    double rho = curand_uniform(rstate);
    double rcount = 0;

    #pragma unroll $REACT_NUM
    for(int react_idx = 0; react_idx < $REACT_NUM; react_idx++)
    {
        rcount = rcount + a[react_idx] / a_0;
        if(rcount >= rho)
        {
            return react_idx;
        }
    }
}

__device__ void UpdateState(int x[$SPECIES_NUM], int j)
{
    // we loop over each non-zero element in the stoichiometry matrix
    #pragma unroll $V_SIZE
    for(int stoich_elem_idx = 0; stoich_elem_idx < $V_SIZE; stoich_elem_idx++)
    {
        char4 stoich_elem = d_V[stoich_elem_idx];
        if(int(stoich_elem.y) == j)
        {
            x[int(stoich_elem.x)] = x[int(stoich_elem.x)] + int(stoich_elem.z);
        }
    }
}

$UPDATE_PROPENSITIES
}
'''
