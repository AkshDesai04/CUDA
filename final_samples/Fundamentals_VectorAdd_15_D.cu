```cuda
/*
 * Aim of the program:
 * Implement a kernel that finds the min and max of a vector in a single pass (requires a parallel reduction, which is more advanced).
 *
 * Thinking process:
 * The task is to compute the minimum and maximum values of a large vector using CUDA in a single traversal of the data.
 * A typical approach is to perform a parallel reduction where each block reduces a chunk of the array into a local min and max.
 * After the kernel finishes, each block writes its local min and max to a small global array.
 * Finally, the host gathers these local results and performs a simple serial reduction to obtain the global min and max.
 *
 * Key design choices:
 * 1. Use a single kernel that performs both min and max reduction concurrently by maintaining two shared arrays (one for min, one for max).
 * 2. Each thread loads one element from the global array; if the index is out of bounds, we initialize the local values with extreme sentinel values (INT_MAX for min, INT_MIN for max).
 * 3. The reduction within a block uses the standard tree-based pattern with a loop over stride values.
 * 4. After intra-block reduction, thread 0 writes the block's min and max into a global output array of size [numBlocks][2].
 * 5. The host copies this small array back and finishes the reduction serially for simplicity.
 *
 * Edge cases handled:
 * - The input size may not be a multiple of the block size; out-of-range indices are ignored by initializing them to sentinels.
 * - The number of elements can be less than the block size; the kernel still works correctly.
 *
 * Performance notes:
 * - Using shared memory significantly reduces global memory traffic during reduction.
 * - The final host reduction is negligible compared to the kernel launch for large vectors.
 *
 * The program includes:
 * - Allocation and initialization of the input vector.
 * - Memory transfers between host and device.
 * - Invocation of the minmax kernel.
 * - Retrieval of results and printing of min and max.
 *
 * Author: ChatGPT
 * Date: 2025-09-02
 */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda.h>

#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

__global__ void minmax_kernel(const int *d_in, int *d_block_results, int N)
{
    extern __shared__ int shared[];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load element or initialize to extreme values if out of bounds
    int val = (global_idx < N) ? d_in[global_idx] : INT_MAX;
    shared[tid] = val; // for min

    val = (global_idx < N) ? d_in[global_idx] : INT_MIN;
    shared[blockDim.x + tid] = val; // for max

    __syncthreads();

    // Reduction for min
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int a = shared[tid];
            int b = shared[tid + stride];
            shared[tid] = (a < b) ? a : b;
        }
        __syncthreads();
    }

    // Reduction for max
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int a = shared[blockDim.x + tid];
            int b = shared[blockDim.x + tid + stride];
            shared[blockDim.x + tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }

    // Thread 0 writes the block's min and max
    if (tid == 0) {
        int block_min = shared[0];
        int block_max = shared[blockDim.x];
        d_block_results[blockIdx.x * 2]     = block_min;
        d_block_results[blockIdx.x * 2 + 1] = block_max;
    }
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    const int THREADS_PER_BLOCK = 256;
    const int BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Allocate host memory
    int *h_in = (int*)malloc(N * sizeof(int));
    if (!h_in) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input with random values
    for (int i = 0; i < N; ++i) {
        h_in[i] = rand() % 1000000 - 500000; // values between -500k and 499k
    }

    // Allocate device memory
    int *d_in = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_in, N * sizeof(int)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate device memory for block results
    int *d_block_results = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_block_results, BLOCKS * 2 * sizeof(int)));

    // Launch kernel
    size_t shared_mem_size = 2 * THREADS_PER_BLOCK * sizeof(int);
    minmax_kernel<<<BLOCKS, THREADS_PER_BLOCK, shared_mem_size>>>(d_in, d_block_results, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy block results back to host
    int *h_block_results = (int*)malloc(BLOCKS * 2 * sizeof(int));
    if (!h_block_results) {
        fprintf(stderr, "Failed to allocate host block results\n");
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_block_results));
        free(h_in);
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaMemcpy(h_block_results, d_block_results, BLOCKS * 2 * sizeof(int), cudaMemcpyDeviceToHost));

    // Final reduction on host
    int global_min = INT_MAX;
    int global_max = INT_MIN;
    for (int i = 0; i < BLOCKS; ++i) {
        int block_min = h_block_results[i * 2];
        int block_max = h_block_results[i * 2 + 1];
        if (block_min < global_min) global_min = block_min;
        if (block_max > global_max) global_max = block_max;
    }

    printf("Global Min: %d\n", global_min);
    printf("Global Max: %d\n", global_max);

    // Clean up
    free(h_in);
    free(h_block_results);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_block_results));

    return 0;
}
```