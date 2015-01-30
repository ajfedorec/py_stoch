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

__device__ __constant__ float d_I[$ITA + 1];  // +1 because I include 0 time point
__device__ __constant__ unsigned char d_E[$KAPPA];

// FUNCTION DECLARATIONS
__device__ void UpdatePropensities(float* a, uint* x, float* c);
__device__ void GetSpecies(uint* temp, uint* x);
__device__ void DetermineCriticalReactions(unsigned char* xeta, uint* x);
__device__ void CalculateMuSigma(uint* x, float* a, unsigned char* xeta,
                                 float* mu, float* sigma2);
__device__ float CalculateTau(unsigned char* xeta, uint* x, float* mu,
                              float* sigma2);
__device__ float CalculateG(uint* x, int i);
__device__ uint SingleCriticalReaction(unsigned char* xeta, float* a,
                                       float a_0_c, curandStateMRG32k3a* rstate);
__device__ void TentativeUpdatedState(int* x_prime, uint* x, int* K);
__device__ bool ValidState(int* x_prime);
__device__ void SaveInterpolatedDynamics(uint* x, int* x_prime, float t0,
                                         float t_Tau, uint f,
                                         uint O[$KAPPA][$ITA][$THREAD_NUM],
                                         int tid);

__global__ void kernel_P1_P2(uint global_x[$THREAD_NUM][$SPECIES_NUM],
                             uint d_O[$KAPPA][$ITA][$THREAD_NUM],
                             int d_Q[$THREAD_NUM],
                             float d_t[$THREAD_NUM],
                             uint d_F[$THREAD_NUM],
                             curandStateMRG32k3a d_rng[$THREAD_NUM])
{
    // 2. tid <- getGlobalId()
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    // 3. sid <- getLocalId()
    int sid = threadIdx.x;

    curandStateMRG32k3a rstate = d_rng[tid];

    // 4. x[sid] <- global_x[tid]
    __shared__ uint x[$BLOCK_SIZE][$SPECIES_NUM];
    for(int species_idx = 0; species_idx < $SPECIES_NUM; species_idx++)
    {
        x[sid][species_idx] = global_x[tid][species_idx];
    }

    // 5. c[sid] <- global_c[tid]
    __shared__ float c[$BLOCK_SIZE][$PARAM_NUM];
    for(int param_idx = 0; param_idx < $PARAM_NUM; param_idx++)
    {
        c[sid][param_idx] = d_c[param_idx];
    }

    // 6. if Q[tid] = -1 then return
    if(d_Q[tid] == -1)
    {
        //printf("Signal of terminated simulation in thread %d\\n", tid);
        return;
    // 7. end if
    }

    // 8. a <- UpdatePropensities( x[sid], c{tid] )
    __shared__ float a[$BLOCK_SIZE][$REACT_NUM];
    for(int react_idx = 0; react_idx < $REACT_NUM; react_idx++)
    {
        a[sid][react_idx] = 0.;
    }
    UpdatePropensities(a[sid], x[sid], c[sid]);

    // 9. a0 <- Sum(a)
    float a_0 = 0.;
    for(int react_idx = 0; react_idx < $REACT_NUM; react_idx++)
    {
        a_0 = a_0 + a[sid][react_idx];
    }

    // 10. if a0 = 0 then
    if(a_0 == 0)
    {
        // 11. for i <- F[tid]...ita do
        for(int species_out_idx = 0; species_out_idx < $KAPPA; species_out_idx++)
        {
            for(int time_out_idx = d_F[tid]; time_out_idx < $ITA;
            time_out_idx++)
            {
                // 13. O[tid][i] <- x[tid]
                d_O[species_out_idx][time_out_idx][tid] = x[sid][d_E[species_out_idx]];
            }
        // 13. end for
        }
        // 14. Q[tid] <- -1
        d_Q[tid] = -1;
        //printf("No reactions can be applied in thread %d\\n", tid);
        // 15. return
        return;
    // 16. end if
    }

    // 17. Xeta <- DetermineCriticalReactions( A, x[sid] )
    __shared__ unsigned char Xeta[$BLOCK_SIZE][$REACT_NUM];
    for(int react_idx = 0; react_idx < $REACT_NUM; react_idx++)
    {
        Xeta[sid][react_idx] = 0;
    }
    DetermineCriticalReactions(Xeta[sid], x[sid]);

    // 18. mu, sigma <- CalculateMuSigma( x[sid], H, H_type, a[sid], Xeta[sid])
    float mu[$A_SIZE];
    float sigma2[$A_SIZE];
    for(int pre_elem_idx = 0; pre_elem_idx < $A_SIZE; pre_elem_idx++)
    {
        mu[pre_elem_idx] = 0.;
        sigma2[pre_elem_idx] = 0.;
    }
    CalculateMuSigma(x[sid], a[sid], Xeta[sid], mu, sigma2);

    // 19. Tau_1 <- CalculateTau(mu, sigma)
    float tau_1 = CalculateTau(Xeta[sid], x[sid], mu, sigma2);
    //printf("tau_1 = %f in thread %d\\n", tau_1, tid);

    // 20. if Tau_1 < 10./a_0 then
    if(tau_1 < 10.0 / a_0)
    {
        // 21. Q[tid] <- 0
        d_Q[tid] = 0;
        //printf("SSA steps are more efficient in thread %d\\n", tid);
        // 22. return
        return;
    }
    // 23. else
    else
    {
        // 24. Q[tid] <- 1
        d_Q[tid] = 1;
        //printf("tau-leaping will be performed in thread %d\\n", tid);
    // 25. end if
    }

    // 26. K <- [], a_0_c <- 0
    __shared__ int K[$BLOCK_SIZE][$REACT_NUM];
    for(int react_idx = 0; react_idx < $REACT_NUM; react_idx++)
    {
        K[sid][react_idx] = 0;
    }
    float a_0_c = 0;

    // 27. for all j in Xeta do
    for(int react_idx = 0; react_idx < $REACT_NUM; react_idx++)
    {
        if(Xeta[sid][react_idx] == 1)
        {
            //printf("reaction %d is critical in thread %d\\n", j, tid);
            // 28. a_0_c <- a_0_c + a[j]  Sum of propensitites of
            // critical reactions
            a_0_c = a_0_c + a[sid][react_idx];
        }
    // 29. end for
    }
    //printf("a0c = %f in thread %d\\n", a_0_c, tid);

    // 30. if a_0_c > 0 then Tau_2 <- (1./a_0_c) * ln(1./rho_1)
    float tau_2 = INFINITY;
    if(a_0_c > 0)
    {
        float rho_1 = curand_uniform(&rstate);
        //printf("rho_1 = %f in thread %d\\n", rho_1, tid);
        tau_2 = (1. / a_0_c) * logf(1. / rho_1);
    // 31. end if
    }
    //printf("tau_2 = %f in thread %d\\n", tau_2, tid);

    // 32. Tau <- min{Tau_1, Tau_2}
    float tau = fminf(tau_1, tau_2);

    // 33. if Tau_1 > Tau_2 then
    if(tau_1 > tau_2)
    {
        // 34. j <- SingleCriticalReaction( Xeta[tid], a[tid] )
        uint j = SingleCriticalReaction(Xeta[sid], a[sid], a_0_c, &rstate);
        //printf("critical reaction %d will be run\\n", j);
        // 35. K[j] <- 1
        K[sid][j] = 1;
    // 36. end if
    }

    __shared__ int x_prime[$BLOCK_SIZE][$SPECIES_NUM];
    for(int species_idx = 0; species_idx < $SPECIES_NUM; species_idx++)
    {
        x_prime[sid][species_idx] = 0;
    }
    // 37. repeat
    while(true)
    {
        // 38. for all j not in Xeta do
        for(int react_idx = 0; react_idx < $REACT_NUM; react_idx++)
        {
            if(Xeta[sid][react_idx] == 0)
            {
                // 39. K[j] <- Poisson( Tau, a[sid][j] )
                float lambda = a[sid][react_idx] * tau;
                K[sid][react_idx] = curand_poisson(&rstate, lambda);
                //printf("reaction %d is NOT CRITICAL, K[%d] = %d in thread %d\\n", j, j, K[j], tid);
            }
        // 40. end for
        }
        // 41. x_prime <- TentativeUpdatedState(x[sid], K[sid], V)
        TentativeUpdatedState(x_prime[sid], x[sid], K[sid]);

        // 42. if ValidState( x_prime ) then break
        if(ValidState(x_prime[sid]))
        {
            //printf("State validated in thread %d\\n", tid);
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
        SaveInterpolatedDynamics(x[sid], x_prime[sid], d_t[tid] - tau, d_t[tid], d_F[tid], d_O, tid);

        // 49. F[tid]++
        d_F[tid] = d_F[tid] + 1;
        //printf("t = %f, d_F = %d, d_I[d_F] = %f\\n", d_t[tid], d_F[tid], d_I[d_F[tid]]);

        // 50. if F[tid] = ita then
        if(d_F[tid] == $ITA)
        {
            // 51. Q[tid] <- -1
            d_Q[tid] = -1;
            //printf("No more samples: simulation over in thread %d\\n", tid);
            break;
        // 52. end if
        }
    // 53. end while
    }
    // 54. global_x <- x_prime
    for(int species_idx = 0; species_idx < $SPECIES_NUM; species_idx++)
    {
        global_x[tid][species_idx] = x_prime[sid][species_idx];
    }
    //printf("time = %f in thread %d\\n", d_t[tid], tid);
// 55. end procedure
}

__device__ void GetSpecies(uint* temp, uint* x)
{
    for(int species_idx = 0; species_idx < $SPECIES_NUM; species_idx++)
    {
        temp[species_idx] = x[d_E[species_idx]];
    }
}

__device__ void DetermineCriticalReactions(unsigned char* xeta, uint* x)
{
    // we loop over each reactant, d_A
    for(int pre_elem_idx = 0; pre_elem_idx < $A_SIZE; pre_elem_idx++)
    {
        // if the stoichiometry of the reactant (d_A[i].z) * our critical
        // reaction threshold (d_n_c) is greater than the current number of
        // molecules of the reactant species (x[d_A[i].x]), the reaction in
        // which the reactant is taking place is classed as critical.
        if((d_A[pre_elem_idx].z * $N_C) > x[d_A[pre_elem_idx].x])
        {
            xeta[d_A[pre_elem_idx].y] = 1;
        }
    }
}

__device__ void CalculateMuSigma(uint* x, float* a, unsigned char* xeta, float* mu,
                                 float* sigma2)
{
    // we loop over each reactant
    for(int pre_elem_idx = 0; pre_elem_idx < $A_SIZE; pre_elem_idx++)
    {
        // we loop over each non-zero element in the stoichiometry matrix
        for(int stoich_elem_idx = 0; stoich_elem_idx < $V_SIZE; stoich_elem_idx++)
        {
            // if the element in the stoichiometry matrix (d_V[v].x) is a
            // reactant (d_A[i].x) in the same reaction (d_V[v].y == d_A[i].y)
            if(d_V[stoich_elem_idx].x == d_A[pre_elem_idx].x)
            {
                if(d_V[stoich_elem_idx].y == d_A[pre_elem_idx].y)
                {
                    // if the reaction (d_V[v].y) is not critical
                    if(xeta[d_V[stoich_elem_idx].y] == 0)
                    {
                        // mu_i = Sum_j_ncr(v_ij * a_j(x)) for all i in {set of
                        // reactant species}
                        // where: v_ij is the stoichiometry of species i in
                        // reaction j
                        //   and a_j(x) is the propensity of reaction j given
                        // state x
                        mu[d_V[stoich_elem_idx].x] = mu[d_V[
                        stoich_elem_idx].x] + d_V[stoich_elem_idx].z * a[d_V[
                        stoich_elem_idx].y];

                        // sigma^2_i = Sum_j_ncr(v_ij^2 * a_j(x)) for all i in {
                        // set of reactant species}
                        sigma2[d_V[stoich_elem_idx].x] = sigma2[d_V[
                        stoich_elem_idx].x] + d_V[stoich_elem_idx].z * d_V[
                        stoich_elem_idx].z * a[d_V[stoich_elem_idx].y];
                    }
                }
            }
        }
    }
}

__device__ float CalculateTau(unsigned char* xeta, uint* x, float* mu, float* sigma2)
{
    // set initial tau to some large number, should be infinite
    float tau  = INFINITY;

    // we loop over each reactant
    for(int pre_elem_idx = 0; pre_elem_idx < $A_SIZE; pre_elem_idx++)
    {
        // if the reactant (d_A[i]) is in a non-critical reaction
        if(xeta[d_A[pre_elem_idx].y] == 0)
        {
            float g_i = CalculateG(x, d_A[pre_elem_idx].x);

            float numerator_l = fmaxf(($ETA * x[d_A[pre_elem_idx].x] / g_i), 1);
            float lhs = numerator_l /  fabsf(mu[d_A[pre_elem_idx].x]);
            float rhs = (numerator_l * numerator_l) / sigma2[d_A[pre_elem_idx].x];
            float temp_tau = fminf(lhs, rhs);

            //printf("d_eta = %f, x[%d] = %d, g_i = %f, mu[%d] = %f, sigma2[%d] = %f\\n", d_eta, d_A[i].x, x[d_A[i].x], g_i, d_A[i].x, mu[d_A[i].x], d_A[i].x, sigma2[d_A[i].x]);
            //printf("numerator_l = %f, lhs = %f, rhs = %f\\n", numerator_l, lhs, rhs);

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
    //printf("d_H[%d] = %d\\n", i, d_H[i]);
    if(d_H[i] == 1)
    {
        return 1.;
    }
    else if(d_H[i] == 2)
    {
        if(d_H_type[i] == 2)
        {
            return (2 + 1 / (x[i] - 1));
        }
        else
        {
            return 2.;
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
            return 3.;
        }
    }
    else
    {
        //printf("Species %d HOR is of order %d and therefore a G value can't be calculated for it.\\n", i, d_H[i]);
        return 0.;
    }
}

__device__ uint SingleCriticalReaction(unsigned char* xeta, float* a, float a_0_c,
                                       curandStateMRG32k3a* rstate)
{
    float rho = curand_uniform(rstate);
    float rcount = 0;

    for(int react_idx = 0; react_idx < $REACT_NUM; react_idx++)
    {
        if(xeta[react_idx] == 1)
        {
            rcount = rcount + a[react_idx] / a_0_c;
            if(rcount >= rho)
            {
                return react_idx;
            }
        }
    }
}

__device__ void TentativeUpdatedState(int* x_prime, uint* x, int* K)
{
    for(int species_idx = 0; species_idx < $SPECIES_NUM; species_idx++)
    {
        x_prime[species_idx] = x[species_idx];
    }

    for(int stoich_elem_idx = 0; stoich_elem_idx < $V_SIZE; stoich_elem_idx++)
    {
        //printf("reaction %d runs %d times with a stoichiometry of %d\\n", d_V[n].y, K[d_V[n].y], d_V[n].z);
        x_prime[d_V[stoich_elem_idx].x] = x_prime[d_V[stoich_elem_idx].x] + K[d_V[stoich_elem_idx].y] * d_V[stoich_elem_idx].z;
    }
}

__device__ bool ValidState(int* x_prime)
{
    for(int species_idx = 0; species_idx < $SPECIES_NUM; species_idx++)
    {
        if(x_prime[species_idx] < 0)
        {
            return false;
        }
    }
    return true;
}

__device__ void SaveInterpolatedDynamics(uint* x, int* x_prime, float t0,
                                         float t_Tau, uint f,
                                         uint O[$KAPPA][$ITA][$THREAD_NUM],
                                         int tid)
{
    float record_time = d_I[f];

    float time_gap = t_Tau - t0;
    float normalised_record_time = record_time - t0;
    float fractional_record_point = normalised_record_time / time_gap;

    //printf("record: %d, t0: %f, t_end: %f, record_time: %f, fractional_record_point: %f\\n", f, t0, t_Tau, record_time, fractional_record_point);

    for(int species_out_idx = 0; species_out_idx < $KAPPA; species_out_idx++)
    {
        //printf("d_E[%d]: %d, xprime[%d]: %d, x[%d]: %d\\n", i, d_E[i], d_E[i], x_prime[d_E[i]], d_E[i], x[d_E[i]]);
        int species_diff = x_prime[d_E[species_out_idx]] - x[d_E[species_out_idx]];
        float fractional_species_increase = species_diff * fractional_record_point;

        uint species_record_point = x[d_E[species_out_idx]] + fractional_species_increase;
        //printf("species[%u] - species_diff: %d, fractional_species_increase: %f, species_record_point: %u\\n", d_E[i], species_diff, fractional_species_increase, species_record_point);

        O[species_out_idx][f][tid] = species_record_point;
    }
}

$UPDATE_PROPENSITIES
}
'''