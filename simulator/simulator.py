import math
import numpy

import pycuda.tools as cuda_tools
from pycuda import characterize
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


class Simulator:
    def CalculateSizes(self, tl_args):
        max_threads_per_block = cuda_tools.DeviceData().max_threads
        max_shared_mem = cuda_tools.DeviceData().shared_memory
        warp_size = cuda_tools.DeviceData().warp_size

        # T <= floor(MAX_shared / (13M + 8N)) from cuTauLeaping paper eq (5)
        threads_per_block = math.floor(
            max_shared_mem / (13 * tl_args.M + 8 * tl_args.N))

        # optimal T is a multiple of warp size
        max_optimal_threads_per_block = min(
            math.floor(threads_per_block / warp_size) * warp_size,
            max_threads_per_block)

        blocks = math.ceil(tl_args.U / 128)


        # grid size is equal to the number of blocks we need
        # grid_size = math.ceil(tl_args.U / optimal_threads_per_block)

        return blocks, 128



class StochasticSimulator(Simulator):
    def get_rng_states(self, size):
        init_rng_src = """
        #include <curand_kernel.h>

        extern "C"
        {

        __global__ void init_rng(int nthreads, curandStateMRG32k3a *s)
        {
            int tid = threadIdx.x + (blockIdx.x * blockDim.x);

            if (tid >= nthreads)
            {
                return;
            }

            curand_init(tid, 0, 0, &s[tid]);
        }

        } // extern "C"
        """

        rng_states = cuda.mem_alloc(
            size * characterize.sizeof('curandStateMRG32k3a',
                                       '#include <curand_kernel.h>'))

        module = SourceModule(init_rng_src, no_extern_c=True)
        init_rng = module.get_function('init_rng')

        init_rng(numpy.int32(size), rng_states, numpy.uint64(0), block=(64, 1, 1),
                 grid=(size // 64 + 1, 1))

        return rng_states