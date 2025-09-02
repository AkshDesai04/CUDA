```cuda
/*********************************************************************
 * Aim of the program:
 *
 * For the final 32 elements (one warp), __syncthreads is not needed
 * between iterations. Use warp shuffle intrinsics (__shfl_down_sync) for
 * a highly optimized final stage of the reduction.
 *
 * ------------------------------------------------------------------
 * Thinking and design decisions:
 *
 * 1. Purpose: Implement a parallel reduction (sum) of an array of
 *    floating‑point numbers on the GPU. The reduction is performed in
 *    three stages:
 *      a) Each thread loads multiple elements from global memory into a
 *         per‑thread accumulator.
 *      b) A shared‑memory reduction collapses the per‑thread accumulators
 *         to a single value per block. For block sizes larger than 32,
 *         the standard shared‑memory reduction with __syncthreads is
 *         performed until only 32 threads remain.
 *      c) The final 32‑thread warp performs the last reduction using
 *         warp shuffle intrinsics (__shfl_down_sync). No __syncthreads
 *         is required because all threads in a warp are already
 *         synchronized by design.
 *
 * 2. Handling arbitrary array size:
 *    The kernel processes the input array in a grid‑stride loop so
 *    that any number of elements can be summed, even if the array size
 *    is not an exact multiple of the block size.
 *
 * 3. Edge cases:
 *    - If the block size is 32 or less, the shared‑memory reduction
 *      step is skipped and the warp shuffle reduction is applied
 *      directly to the per‑thread sums.
 *    - If the array size is smaller than the grid size, many blocks
 *      will produce zero partial sums; these are harmless for the
 *      final host‑side accumulation.
 *
 * 4. Final host reduction:
 *    After launching the kernel, the host gathers the partial sums
 *    produced by each block and reduces them on the CPU to obtain the
 *    final result. This keeps the example simple while still
 *    demonstrating the optimized warp‑level reduction on the GPU.
 *
 * 5. Miscellaneous:
 *    - We use a 32‑bit unsigned mask (0xffffffff) for the shuffle
 *      intrinsic, which is the full‑warp mask. In practice, one could
 *      use __activemask() if the warp size might be reduced.
 *    - The kernel is written for CUDA 9.0+ where __shfl_down_sync
 *      is available. For older devices, the older __shfl_down()
 *      intrinsic could be used instead.
 *
 * ------------------------------------------------------------------
 * End of comment.  The following is the complete .cu source file.
 *********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Simple CUDA error checking macro
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel: per‑block reduction using shared memory + warp shuffle
__global__ void reduceSum(const float *input, float *blockSums, int N)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    unsigned int stride = gridDim.x * blockDim.x;

    // 1. Load elements from global memory into per‑thread accumulator
    float sum = 0.0f;
    for (int i = idx; i < N; i += stride) {
        sum += input[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // 2. Shared‑memory reduction
    // If blockDim.x > 32, reduce until only 32 threads remain
    unsigned int blockSize = blockDim.x;
    if (blockSize > 32) {
        // Standard reduction with __syncthreads
        for (unsigned int s = blockSize >> 1; s > 32; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        // Now blockSize <= 32, fall through to warp shuffle
    }

    // 3. Warp‑level reduction using shuffle intrinsics (no __syncthreads)
    // Only threads with tid < 32 participate
    float val = sdata[tid];
    // If the block had less than 32 threads, the above loop was skipped
    // and sdata[tid] is already correct for all active threads.
    // We now perform shuffle reduction on the 32 (or fewer) active threads.
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // 4. Store the block's partial sum in global memory
    if (tid == 0) {
        blockSums[blockIdx.x] = val;
    }
}

// Host helper to launch the kernel and perform final reduction
float reduceSumHost(const float *h_in, int N)
{
    // Device pointers
    float *d_in = nullptr;
    float *d_blockSums = nullptr;

    // Allocate device memory for input
    CHECK_CUDA(cudaMalloc((void**)&d_in, N * sizeof(float)));
    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // Determine block and grid sizes
    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate device memory for block sums
    CHECK_CUDA(cudaMalloc((void**)&d_blockSums, blocks * sizeof(float)));

    // Launch kernel
    size_t sharedMemSize = threadsPerBlock * sizeof(float);
    reduceSum<<<blocks, threadsPerBlock, sharedMemSize>>>(d_in, d_blockSums, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy block sums back to host
    float *h_blockSums = (float*)malloc(blocks * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_blockSums, d_blockSums, blocks * sizeof(float), cudaMemcpyDeviceToHost));

    // Final reduction on host
    float total = 0.0f;
    for (int i = 0; i < blocks; ++i) {
        total += h_blockSums[i];
    }

    // Clean up
    free(h_blockSums);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_blockSums));

    return total;
}

int main()
{
    const int N = 1 << 20; // 1M elements
    float *h_in = (float*)malloc(N * sizeof(float));

    // Initialize input with some values (e.g., 1.0f)
    for (int i = 0; i < N; ++i) {
        h_in[i] = 1.0f; // Simple test case
    }

    // Compute sum using the CUDA reduction
    float gpuSum = reduceSumHost(h_in, N);

    // Compute sum on CPU for verification
    float cpuSum = 0.0f;
    for (int i = 0; i < N; ++i) {
        cpuSum += h_in[i];
    }

    printf("GPU sum: %.6f\n", gpuSum);
    printf("CPU sum: %.6f\n", cpuSum);
    printf("Difference: %.6e\n", fabsf(gpuSum - cpuSum));

    free(h_in);
    return 0;
}
```