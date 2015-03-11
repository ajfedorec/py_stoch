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
__device__ __constant__ int d_E[$KAPPA];  // output species indexes

// FUNCTION DECLARATIONS
__device__ void UpdatePropensities(double a[$REACT_NUM],
                                   int x[$SPECIES_NUM]);

__device__ void DetermineCriticalReactions(int xeta[$REACT_NUM],
                                           int x[$SPECIES_NUM]);

__device__ void CalculateMuSigma(int x[$SPECIES_NUM],
                                 double a[$REACT_NUM],
                                 int xeta[$REACT_NUM],
                                 double mu[$SPECIES_NUM],
                                 double sigma2[$SPECIES_NUM]);

__device__ double CalculateTau(int xeta[$REACT_NUM],
                               int x[$SPECIES_NUM],
                               double mu[$SPECIES_NUM],
                               double sigma2[$SPECIES_NUM]);

__device__ double CalculateG(int x_idx, int i);

__device__ int SingleCriticalReaction(int xeta[$REACT_NUM],
                                      double a[$REACT_NUM],
                                      double a_0_c,
                                      curandStateMRG32k3a* rstate);

__device__ void TentativeUpdatedState(int x_prime[$SPECIES_NUM],
                                      int x[$SPECIES_NUM],
                                      int K[$REACT_NUM]);

__device__ bool ValidState(int x_prime[$SPECIES_NUM]);

__device__ void SaveInterpolatedDynamics(int x[$SPECIES_NUM],
                                         int x_prime[$SPECIES_NUM],
                                         double t0,
                                         double t_Tau,
                                         int f,
                                         int O[$KAPPA][$ITA][$THREAD_NUM],
                                         int tid);

__global__ void kernel_P1_P2(int global_x[$THREAD_NUM][$SPECIES_NUM],
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

    // Uncomment this to force gillespie
    //if(d_Q[tid] != -1)
    //{
    //    d_Q[tid] = 0;
    //    return;
    //}

    __shared__ curandStateMRG32k3a rstate[$BLOCK_SIZE];
    rstate[sid] = d_rng[tid];

    // 4. x[sid] <- global_x[tid]
    __shared__ int x[$BLOCK_SIZE][$SPECIES_NUM];
    for(int species_idx = 0; species_idx < $SPECIES_NUM; species_idx++)
    {
        x[sid][species_idx] = global_x[tid][species_idx];
		//printf("TL: x[%d] = %d at time %f in thread %d\\n", species_idx,
		//        x[sid][species_idx], d_t[tid], tid);
    }

    // 6. if Q[tid] = -1 then return
    if(d_Q[tid] == -1)
    {
        //printf("Signal of terminated simulation in thread %d\\n", tid);
        return;
    // 7. end if
    }

    // 8. a <- UpdatePropensities( x[sid], c{tid] )
    __shared__ double a[$BLOCK_SIZE][$REACT_NUM];
    for(int react_idx = 0; react_idx < $REACT_NUM; react_idx++)
    {
        a[sid][react_idx] = 0.;
    }
    UpdatePropensities(a[sid], x[sid]);

    // 9. a0 <- Sum(a)
    double a_0 = 0.;
    for(int react_idx = 0; react_idx < $REACT_NUM; react_idx++)
    {
        a_0 = a_0 + a[sid][react_idx];
    }

    // 10. if a0 = 0 then
    if(a_0 == 0)
    {
        // 11. for i <- F[tid]...ita do
        for(int species_out_idx = 0; species_out_idx < $KAPPA;
            species_out_idx++)
        {
            for(int time_out_idx = d_F[tid]; time_out_idx < $ITA;
                time_out_idx++)
            {
                // 13. O[tid][i] <- x[tid]
                d_O[species_out_idx][time_out_idx][tid] =
                                                   x[sid][d_E[species_out_idx]];
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
    __shared__ int Xeta[$BLOCK_SIZE][$REACT_NUM];
    for(int react_idx = 0; react_idx < $REACT_NUM; react_idx++)
    {
        Xeta[sid][react_idx] = 0;
    }
    DetermineCriticalReactions(Xeta[sid], x[sid]);

    // 18. mu, sigma <- CalculateMuSigma( x[sid], H, H_type, a[sid], Xeta[sid])
    __shared__ double mu[$BLOCK_SIZE][$SPECIES_NUM];
    __shared__ double sigma2[$BLOCK_SIZE][$SPECIES_NUM];
    for(int species_idx = 0; species_idx < $SPECIES_NUM; species_idx++)
    {
        mu[sid][species_idx] = 0.;
        sigma2[sid][species_idx] = 0.;
    }
    CalculateMuSigma(x[sid], a[sid], Xeta[sid], mu[sid], sigma2[sid]);

    // 19. Tau_1 <- CalculateTau(mu, sigma)
    double tau_1 = CalculateTau(Xeta[sid], x[sid], mu[sid], sigma2[sid]);
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
    double a_0_c = 0.;

    // 27. for all j in Xeta do
    for(int react_idx = 0; react_idx < $REACT_NUM; react_idx++)
    {
        if(Xeta[sid][react_idx] == 1)
        {
            //printf("reaction %d is critical in thread %d\\n", react_idx, tid);
            // 28. a_0_c <- a_0_c + a[j]  Sum of propensitites of
            // critical reactions
            a_0_c = a_0_c + a[sid][react_idx];
        }
    // 29. end for
    }
    //printf("a0c = %f in thread %d\\n", a_0_c, tid);

    // 30. if a_0_c > 0 then Tau_2 <- (1./a_0_c) * ln(1./rho_1)
    double tau_2 = INFINITY;
    if(a_0_c > 0.)
    {
        double rho_1 = curand_uniform(&rstate[sid]);
        //printf("rho_1 = %f in thread %d\\n", rho_1, tid);
        tau_2 = (1. / a_0_c) * log(1. / rho_1);
    // 31. end if
    }
    ///printf("tau_2 = %f in thread %d\\n", tau_2, tid);

    // 32. Tau <- min{Tau_1, Tau_2}
    double tau = fminf(tau_1, tau_2);

    // 33. if Tau_1 > Tau_2 then
    if(tau_1 > tau_2)
    {
        // 34. j <- SingleCriticalReaction( Xeta[tid], a[tid] )
        int j = SingleCriticalReaction(Xeta[sid], a[sid], a_0_c, &rstate[sid]);
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

    double lambda = 0.;
    // 37. repeat
    while(true)
    {
        // 38. for all j not in Xeta do
        for(int react_idx = 0; react_idx < $REACT_NUM; react_idx++)
        {
            if(Xeta[sid][react_idx] == 0)
            {
                // 39. K[j] <- Poisson( Tau, a[sid][j] )
                lambda = a[sid][react_idx] * tau;
                K[sid][react_idx] = curand_poisson(&rstate[sid], lambda);
                //printf("reaction %d is NOT CRITICAL, K[%d] = %d in thread %d\\n",
                //        react_idx, react_idx, K[sid][react_idx], tid);
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
        tau = tau / 2.;
    // 45. until False
    }

    // 46. t[tid] <- t[tid] + Tau
    d_t[tid] = d_t[tid] + tau;

    // 47. while t[tid] >= I[F[tid]] do
    while(d_t[tid] >= d_I[d_F[tid]])
    {
        // 48. SaveInterpolatedDynamics(x, x_prime, O[tid], F[tid])
        SaveInterpolatedDynamics(x[sid], x_prime[sid], d_t[tid] - tau,
                                 d_t[tid], d_F[tid], d_O, tid);

        // 49. F[tid]++
        d_F[tid] = d_F[tid] + 1;
        //printf("t = %f, d_F = %d, d_I[d_F] = %f\\n", d_t[tid], d_F[tid],
        //        d_I[d_F[tid]]);

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
        //printf("TL: global_x[%d][%d] = %d at time %f\\n", tid, species_idx,
        //        global_x[tid][species_idx], d_t[tid]);
    }
    d_rng[tid] = rstate[sid];
    //printf("time = %f in thread %d\\n", d_t[tid], tid);
// 55. end procedure
}

__device__ void DetermineCriticalReactions(int xeta[$REACT_NUM],
                                           int x[$SPECIES_NUM])
{
    //// we loop over each reactant, d_A
    //for(int pre_elem_idx = 0; pre_elem_idx < $A_SIZE; pre_elem_idx++)
    //{
        //// if the stoichiometry of the reactant (d_A[i].z) * our critical
        //// reaction threshold (d_n_c) is greater than the current number of
        //// molecules of the reactant species (x[d_A[i].x]), the reaction in
        //// which the reactant is taking place is classed as critical.
        //if((d_A[pre_elem_idx].z * $N_C) > x[d_A[pre_elem_idx].x])
        //{
            //xeta[d_A[pre_elem_idx].y] = 1;
        //}
    //}
    char4 stoich_elem;
    for(int stoich_elem_idx = 0; stoich_elem_idx < $V_SIZE; stoich_elem_idx++)
    {
        stoich_elem = d_V[stoich_elem_idx];
        if(int(stoich_elem.z) < 0)
        {
            // this line is a hack to do a quick abs(x) for integers
            int abs_stoich = int(stoich_elem.z) < 0 ? -int(stoich_elem.z) : int(stoich_elem.z);
            if((abs_stoich * $N_C) > x[int(stoich_elem.x)])
            {
                xeta[int(stoich_elem.y)] = 1;
            }
        }
    }
}

__device__ void CalculateMuSigma(int x[$SPECIES_NUM],
                                 double a[$REACT_NUM],
                                 int xeta[$REACT_NUM],
                                 double mu[$SPECIES_NUM],
                                 double sigma2[$SPECIES_NUM])
{
    // need to calculate I_rs, a list of species appearing as a reactant in at
    // least one reaction
    int I_rs[$SPECIES_NUM];
    for(int species_idx = 0; species_idx < $SPECIES_NUM; species_idx++)
    {
        I_rs[species_idx] = 0;
    }
    for(int pre_elem_idx = 0; pre_elem_idx < $A_SIZE; pre_elem_idx++)
    {
        I_rs[int(d_A[pre_elem_idx].x)] = 1;
    }

    char4 stoich_elem;

    // we loop over each non-zero element in the stoichiometry matrix
    for(int stoich_elem_idx = 0; stoich_elem_idx < $V_SIZE; stoich_elem_idx++)
    {
        stoich_elem = d_V[stoich_elem_idx];

        //printf("species: %d, react: %d, reactant?: %d, crit?: %d\\n",
        //        stoich_elem.x, stoich_elem.y, I_rs[stoich_elem.x],
        //        xeta[stoich_elem.y]);

        if(I_rs[int(stoich_elem.x)] == 1)
        {
            // if the reaction (stoich_elem.y) is not critical
            if(xeta[int(stoich_elem.y)] == 0)
            {
                // mu_i = Sum_j_ncr(v_ij * a_j(x)) for all i in {set of
                // reactant species}
                // where: v_ij is the stoichiometry of species i in
                // reaction j
                //   and a_j(x) is the propensity of reaction j given
                // state x
                mu[int(stoich_elem.x)] = mu[int(stoich_elem.x)] + (int(stoich_elem.z) *
                                    a[int(stoich_elem.y)]);

                // sigma^2_i = Sum_j_ncr(v_ij^2 * a_j(x)) for all i in {
                // set of reactant species}
                sigma2[int(stoich_elem.x)] = sigma2[int(stoich_elem.x)] +
                                        ((int(stoich_elem.z) * int(stoich_elem.z)) *
                                        a[int(stoich_elem.y)]);
            }
        }
    }
}

__device__ double CalculateTau(int xeta[$REACT_NUM],
                               int x[$SPECIES_NUM],
                               double mu[$SPECIES_NUM],
                               double sigma2[$SPECIES_NUM])
{
    // set initial tau to some large number, should be infinite
    double tau  = INFINITY;

    // need to calculate I_rs, a list of species appearing as a reactant in at
    // least one reaction
    int I_rs[$SPECIES_NUM];
    for(int species_idx = 0; species_idx < $SPECIES_NUM; species_idx++)
    {
        I_rs[species_idx] = 0;
    }
    for(int pre_elem_idx = 0; pre_elem_idx < $A_SIZE; pre_elem_idx++)
    {
        I_rs[int(d_A[pre_elem_idx].x)] = 1;
    }

    char4 stoich_elem;
    // we loop over each non-zero element in the stoichiometry matrix
    for(int stoich_elem_idx = 0; stoich_elem_idx < $V_SIZE; stoich_elem_idx++)
    {
        stoich_elem = d_V[stoich_elem_idx];

        if(I_rs[int(stoich_elem.x)] == 1)
        {

            // if the reactant (d_A[i]) is in a non-critical reaction
            if(xeta[int(stoich_elem.y)] == 0)
            {
                double g_i = CalculateG(x[int(stoich_elem.x)], int(stoich_elem.x));

                double numerator_l = fmaxf(($ETA * x[int(stoich_elem.x)] / g_i), 1.);
                double lhs = numerator_l /  fabsf(mu[int(stoich_elem.x)]);
                double rhs = (numerator_l * numerator_l) / sigma2[int(stoich_elem.x)];
                double temp_tau = fminf(lhs, rhs);

                //printf("x[%d] = %d, g_i = %f, mu[%d] = %f, sigma2[%d] =%f\\n",
                //        stoich_elem.x, x[stoich_elem.x], g_i, stoich_elem.x,
                //        mu[stoich_elem.x], stoich_elem.x,
                //        sigma2[stoich_elem.x]);
                //printf("numerator_l = %f, lhs = %f, rhs = %f\\n",
                //        numerator_l, lhs, rhs);

                //printf("character %c, integer %d, cast int %d\\n", stoich_elem.x, stoich_elem.x, int(stoich_elem.x));

                if(temp_tau < tau)
                {
                    tau = temp_tau;
                }
            }
        }
    }
    return tau;
}

__device__ double CalculateG(int x_i, int i)
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
            return (2. + 1. / (x_i - 1.));
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
            return (3. + 1. / (x_i - 1.) + 2. / (x_i - 2.));
        }
        else if(d_H_type[i] == 2)
        {
            return 3. / 2. * (2. + 1. / (x_i - 1.));
        }
        else
        {
            return 3.;
        }
    }
    else
    {
        //printf("Species %d HOR is of order %d and therefore a G value can't
        //        be calculated for it.\\n", i, d_H[i]);
        return 0.;
    }
}

__device__ int SingleCriticalReaction(int xeta[$REACT_NUM],
                                      double a[$REACT_NUM],
                                      double a_0_c,
                                      curandStateMRG32k3a* rstate)
{
    double rho = curand_uniform(rstate);
    double rcount = 0.;

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

__device__ void TentativeUpdatedState(int x_prime[$SPECIES_NUM],
                                      int x[$SPECIES_NUM],
                                      int K[$REACT_NUM])
{
    for(int species_idx = 0; species_idx < $SPECIES_NUM; species_idx++)
    {
        x_prime[species_idx] = x[species_idx];
    }

    char4 stoich_elem;
    for(int stoich_elem_idx = 0; stoich_elem_idx < $V_SIZE; stoich_elem_idx++)
    {
        stoich_elem = d_V[stoich_elem_idx];
        //printf("reaction %d runs %d times with a stoichiometry of %d\\n",
        //        d_V[n].y, K[d_V[n].y], d_V[n].z);
        x_prime[int(stoich_elem.x)] = x_prime[int(stoich_elem.x)] +
                                          K[int(stoich_elem.y)] *
                                          int(stoich_elem.z);
    }
}

__device__ bool ValidState(int x_prime[$SPECIES_NUM])
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

__device__ void SaveInterpolatedDynamics(int x[$SPECIES_NUM],
                                         int x_prime[$SPECIES_NUM],
                                         double t0,
                                         double t_Tau,
                                         int f,
                                         int O[$KAPPA][$ITA][$THREAD_NUM],
                                         int tid)
{
    float record_time = d_I[f];

    double time_gap = t_Tau - t0;
    double normalised_record_time = record_time - t0;
    double fractional_record_point = normalised_record_time / time_gap;

    //printf("record: %d, t0: %f, t_end: %f, record_time: %f,
    //        fractional_record_point: %f\\n", f, t0, t_Tau, record_time,
    //        fractional_record_point);

    for(int species_out_idx = 0; species_out_idx < $KAPPA; species_out_idx++)
    {
        //printf("d_E[%d]: %d, xprime[%d]: %d, x[%d]: %d\\n",
        //        species_out_idx, d_E[species_out_idx], d_E[species_out_idx],
        //        x_prime[d_E[species_out_idx]], d_E[species_out_idx],
        //        x[d_E[species_out_idx]]);
        int species_diff = x_prime[d_E[species_out_idx]] -
                           x[d_E[species_out_idx]];
        double fractional_species_increase = species_diff *
                                            fractional_record_point;

        int species_record_point = x[d_E[species_out_idx]] +
                                    __float2int_rn(fractional_species_increase);
        //printf("species[%u] - species_diff: %d, fractional_species_increase:
        //        %f, species_record_point: %u\\n", d_E[species_out_idx],
        //        species_diff, fractional_species_increase,
        //        species_record_point);

        //printf("i: %d, d_E[%d]: %d, x[%d]: %d, f: %d\\n", species_out_idx,
        //        species_out_idx, d_E[species_out_idx], d_E[species_out_idx],
        //        x[d_E[species_out_idx]], f);

        O[species_out_idx][f][tid] = species_record_point;
    }
}

$UPDATE_PROPENSITIES
}
'''
