kernel = '''
__global__ void kernel_P3()
{
    // 2. tid <- getGloablId()
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    // 3. sid <- getLocalId()
    int sid = threadIdx.x;

    // 4. if Q[tid] != 0 then return
    if(d_Q[tid] != 0)
    {
        return;
    // 5. end if
    }

    // 6. x[sid] <- global_x[tid]
    uint x[$SPECIES_NUM];
    for(int i = 0; i < d_N; i++)
    {
        x[i] = d_x[tid][i];
    }

    // 7. c[sid] <- global_c[tid]
    float c[$PARAM_NUM];
    for(int i = 0; i < $PARAM_NUM; i++)
    {
        c[i] = d_c[i];
    }

    // 8. for i in 0...100 do
    for(int i = 0; i < 100; i++)
    {
        // 9. a[sid] <- UpdatePropensities( x[sid], c[sid] )
        float a[$REACT_NUM];
        UpdatePropensities(a, x, c);

        // 10. a_0 <- Sum(a[sid])
        float a_0 = 0.;
        for(int j = 0; j < $REACT_NUM; j++)
        {
            a_0 = a_0 + a[j];
        }

        // 11. if a_0 = 0 then
        if(a_0 == 0)
        {
            // 12. for i <- E[tid]...ita do
            for(int k = d_E[tid]; k < d_ita; k++)
            {
                // 13. O[tid][i] <- x[tid]
                for(int n = 0; n < d_N; n++)
                {
                    d_O[n][i][tid] = x[n];
                }
            // 14. end for
            }
            // 15. Q[tid] <- -1
            d_Q[tid] = -1;
            printf("No reactions can be applied in thread %d\\n", tid);
            // 16. return
            return;
        // 17. end if
        }

        // 18. Tau <- (1./a_0) * ln(1./rho_1)
        curandStateMRG32k3a rstate;
        curand_init(0, tid, 0, &rstate);
        float tau = 1000000.;
        float rho_1 = curand_uniform(&rstate);
        tau = (1. / a_0) * logf(1. / rho_1);

        // 19. j <- SingleCriticalReaction( xeta[sid], a[sid] )
        //*****TODO*****

        // 20. T[tid] <- T[tid] + Tau
        d_t[tid] = d_t[tid] + tau;

        // 21. if T[tid] >= I[F[tid]] then
        if(d_t[tid] >= d_I[d_F[tid]])
        {
            // 22. SaveDynamics(x[sid],O[tid],E[tid])
            //*****TODO*****

            //23. F[tid]++
            d_F[tid] = d_F[tid] + 1;

            // 24. if F[tid] = ita then
            if(d_F[tid] == d_ita)
            {
                // 25. Q[tid] <- -1
                d_Q[tid] = -1;
                printf("No more samples: simulation over in thread %d\\n", tid);
            // 26. end if
            }
        // 27. end if
        }
        // 28. x <- UpdateState( x, j )
        //*****TODO*****
    // 29. end for
    }
    // 30. global_x <- x
    for(int i = 0; i < d_N; i++)
    {
        d_x[tid][i] = x[i];
    }
// 31. end procedure
}
'''

