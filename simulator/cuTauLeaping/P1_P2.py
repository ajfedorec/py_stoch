kernel = '''
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
__device__ float d_I[$ITA];
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
__device__ uint SingleCriticalReaction(uint* xeta, float* a, float a_0_c,
curandStateMRG32k3a* rstate);
__device__ void TentativeUpdatedState(int* x_prime, uint* x, float* K);
__device__ bool ValidState(int* x_prime);

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
    uint x[$SPECIES_NUM];
    for(int i = 0; i < d_N; i++)
    {
        x[i] = d_x[tid][i];
    }

    // 5. c[sid] <- global_c[tid]
    float c[$PARAM_NUM];
    for(int i = 0; i < $PARAM_NUM; i++)
    {
        c[i] = d_c[i];
    }

    // 6. if Q[tid] = -1 then return
    if(d_Q[tid] == -1)
    {
        printf("Signal of terminated simulation in thread %d\\n", tid);
        return;
    // 7. end if
    }

    // 8. a <- UpdatePropensities( x[sid], c{tid] )
    float a[$REACT_NUM];
    UpdatePropensities(a, x, c);

    // 9. a0 <- Sum(a)
    float a_0 = 0.;
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
            // 12. O[tid][i] <- GetSpecies(x[sid], E)
            GetSpecies(&d_O[0][tid][i], x, d_E);
        // 13. end for
        }
        // 14. Q[tid] <- -1
        d_Q[tid] = -1;
        printf("No reactions can be applied in thread %d\\n", tid);
        // 15. return
        return;
    // 16. end if
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
        // 21. Q[tid] <- 0
        d_Q[tid] = 0;
        printf("SSA steps are more efficient in thread %d\\n", tid);
        // 22. return
        return;
    }
    // 23. else
    else
    {
        // 24. Q[tid] <- 1
        d_Q[tid] = 1;
        printf("tau-leaping will be performed in thread %d\\n", tid);
    // 25. end if
    }

    // 26. K <- [], a_0_c <- 0
    float K[$REACT_NUM];
    float a_0_c = 0;

    // 27. for all j in Xeta do
    for(int j = 0; j < d_M; j++)
    {
        if(Xeta[j] == 1)
        {
            // 28. a_0_c <- a_0_c + a[j]  Sum of propensitites of
            // critical reactions
            a_0_c = a_0_c + a[j];
        }
    // 29. end for
    }

    // 30. if a_0_c > 0 then Tau_2 <- (1./a_0_c) * ln(1./rho_1)
    curandStateMRG32k3a rstate;
    curand_init(0, tid, 0, &rstate);
    float tau_2 = 1000000.;
    if(a_0_c > 0)
    {
        float rho_1 = curand_uniform(&rstate);
        tau_2 = (1. / a_0_c) * logf(1. / rho_1);
    // 31. end if
    }

    // 32. Tau <- min{Tau_1, Tau_2}
    float tau = fminf(tau_1, tau_2);

    // 33. if Tau_1 > Tau_2 then
    if(tau_1 > tau_2)
    {
        // 34. j <- SingleCriticalReaction( Xeta[tid], a[tid] )
        uint j = SingleCriticalReaction(Xeta, a, a_0_c, &rstate);
        // 35. K[j] <- 1
        K[j] = 1;
    // 36. end if
    }

    int x_prime[$SPECIES_NUM];
    // 37. repeat
    while(true)
    {
        // 38. for all j not in Xeta do
        for(int j = 0; j < d_M; j++)
        {
            if(Xeta[j] == 0)
            {
                // 39. K[j] <- Poisson( Tau, a[sid][j] )
                float lambda = a[j] * tau;
                K[j] = curand_poisson(&rstate, lambda);
            }
        // 40. end for
        }
        // 41. x_prime <- TentativeUpdatedState(x[sid], K[sid], V)
        TentativeUpdatedState(x_prime, x, K);

        // 42. if ValidState( x_prime ) then break
        if(ValidState(x_prime))
        {
            break;
        // 43. end if
        }

        // 44. Tau <- Tau / 2
        tau = tau / 2;
    // 45. until False
    }

    // 46. t[tid] <- t[tid] + Tau
    d_t[tid] = d_t[tid] + tau;

    // 47. while t[tid] >= I[F[tid]] do
    while(d_t[tid] >= d_I[d_F[tid]])
    {
        // 48. SaveInterpolatedDynamics(x, x_prime, O[tid], F[tid])
        //*****TODO*****

        // 49. F[tid]++
        d_F[tid] = d_F[tid] + 1;

        // 50. if F[tid] = ita then
        if(d_F[tid] == d_ita)
        {
            // 51. Q[tid] <- -1
            d_Q[tid] = -1;
            printf("No more samples: simulation over in thread %d\\n", tid);
        // 52. end if
        }
    // 53. end while
    }
    // 54. global_x <- x_prime
    for(int i = 0; i < d_N; i++)
    {
        d_x[tid][i] = x_prime[i];
    }
// 55. end procedure
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
                    // reactant species}
                    // where: v_ij is the stoichiometry of species i in
                    // reaction j
                    //   and a_j(x) is the propensity of reaction j given
                    // state x
                    mu[d_V[v].x] = mu[d_V[v].x] + d_V[v].z * a[d_V[v].y];

                    // sigma^2_i = Sum_j_ncr(v_ij^2 * a_j(x)) for all i in {
                    // set of reactant species}
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

__device__ uint SingleCriticalReaction(uint* xeta, float* a, float a_0_c,
curandStateMRG32k3a* rstate)
{
    float rho = curand_uniform(rstate);
    float rcount = 0;

    for(int j = 0; j < d_M; j++)
    {
        if(xeta[j] == 1)
        {
            rcount = rcount + a[j] / a_0_c;
            if(rcount >= rho)
            {
                return j;
            }
        }
    }
}

__device__ void TentativeUpdatedState(int* x_prime, uint* x, float* K)
{
    for(int i = 0; i < d_N; i++)
    {
        x_prime[i] = x[i];
    }

    for(int n = 0; n < d_V_size; n++)
    {
        x_prime[d_V[n].x] = x_prime[d_V[n].x] + K[d_V[n].y] * d_V[n].z;
    }
}

__device__ bool ValidState(int* x_prime)
{
    for(int i = 0; i < d_N; i++)
    {
        if(x_prime[i] < 0)
        {
            return false;
        }
    }
    return true;
}

$UPDATE_PROPENSITIES
}
'''