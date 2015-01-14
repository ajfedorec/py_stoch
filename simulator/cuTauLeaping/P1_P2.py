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
    uint* x[sid] = global_x[tid];

    // c[sid] <- global_c[tid]
    float* c[sid] = global_c[tid];

    // if Q[tid] = -1 then return
    if Q[tid] == -1
    {
        return;
    }

    printf("%d", tid);
'''
