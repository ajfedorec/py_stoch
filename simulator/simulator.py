import math
import numpy

import pycuda.tools as cuda_tools
from pycuda import characterize
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


class Simulator:
    def CalculateSizes(self, tl_args):
        hw_constrained_threads_per_block = cuda_tools.DeviceData().max_threads

        # T <= floor(MAX_shared / (13M + 8N)) from cuTauLeaping paper eq (5)
        # threads_per_block = math.floor(
        # max_shared_mem / (13 * tl_args.M + 8 * tl_args.N))
        # HOWEVER, for my implementation:
        #   type                size    var     number
        #   curandStateMRG32k3a 80      rstate  1
        #   uint                32      x       N
        #   float               32      c       P
        #   float               32      a       M
        #   u char              8       Xeta    M
        #   int                 32      K       M
        #   int                 32      x_prime N
        #   T <= floor(Max_shared / (9M + 8N + 4P + 10) (bytes)
        max_shared_mem = cuda_tools.DeviceData().shared_memory
        shared_mem_constrained_threads_per_block = math.floor(max_shared_mem / (
        9 * tl_args.M + 8 * tl_args.N + 4 * len(tl_args.c)) + 10)

        max_threads_per_block = min(hw_constrained_threads_per_block,
                                    shared_mem_constrained_threads_per_block)

        warp_size = cuda_tools.DeviceData().warp_size

        # optimal T is a multiple of warp size
        max_warps_per_block = math.floor(max_threads_per_block / warp_size)
        max_optimal_threads_per_block = max_warps_per_block * warp_size

        if max_optimal_threads_per_block >= 256:
            block_size = 256
        elif max_optimal_threads_per_block >= 128:
            block_size = 128
        else:
            block_size = max_optimal_threads_per_block

        if tl_args.U <= 2:
            block_size = tl_args.U

        grid_size = int(math.ceil(float(tl_args.U) / float(block_size)))

        tl_args.U = int(grid_size * block_size)

        return grid_size, block_size


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

        init_rng(numpy.int32(size), rng_states, numpy.uint64(0),
                 block=(64, 1, 1),
                 grid=(size // 64 + 1, 1))

        return rng_states