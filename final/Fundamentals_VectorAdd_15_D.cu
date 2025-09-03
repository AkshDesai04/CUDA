/*
Aim: Implement a kernel that finds the min and max of a vector in a single pass (requires a parallel reduction, which is more advanced).

Thinking:
The problem requires us to compute the minimum and maximum values of a large vector using CUDA.  The key requirement is that the data should be read only once – we call this a "single pass".  The common way to achieve this is to use a parallel reduction: each thread block reduces its chunk of the array into a local min and max, and then the results of all blocks are combined into the global min and max.

Design choices:

1. **Block‑level reduction**  
   - Each block will load its portion of the input into shared memory.  
   - We keep two separate shared arrays: one for the current minimums and one for the current maximums.  
   - A standard tree‑based reduction is used: starting with stride = blockDim.x/2 and halving each iteration, each thread compares its element with the partner element and keeps the min or max.

2. **Handling bounds**  
   - Because the vector length may not be a multiple of the block size, each thread checks whether its global index is within bounds before loading.  
   - If an index is out of bounds, the thread writes a sentinel value (`FLT_MAX` for min and `-FLT_MAX` for max) so that it does not affect the reduction.

3. **Storing block results**  
   - After the reduction, thread 0 of each block writes the block’s local min and max to global arrays `blockMin` and `blockMax`.  
   - These arrays have a size equal to the number of blocks.

4. **Final reduction**  
   - After the kernel finishes, the host copies `blockMin` and `blockMax` back to host memory.  
   - The host performs a simple linear scan over these arrays to compute the final global min and max.  
   - This two‑stage approach still respects the “single pass” requirement because the original vector is read only once in the kernel.

5. **Performance considerations**  
   - The kernel uses `__shared__` memory which is the fastest memory available on a GPU for intra‑block communication.  
   - All threads in a warp participate in each reduction step, ensuring good occupancy.  
   - We do not use atomic operations in the kernel, keeping it highly parallel.

6. **Testing**  
   - The main function allocates a large vector (e.g., 1<<20 elements), fills it with random float values, and calls the kernel.  
   - The computed min and max are printed and compared with a CPU reference implementation to verify correctness.

This implementation follows best practices for CUDA reduction patterns and meets the requirement of a single pass over the data. */

#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <time.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",     \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Kernel that reduces a chunk of the input array into per-block min and max
__global__ void minMaxKernel(const float *input, float *blockMin, float *blockMax, int N)
{
    // Shared memory for min and max
    extern __shared__ float sdata[]; // size = 2 * blockDim.x
    float *smin = sdata;            // first blockDim.x entries
    float *smax = sdata + blockDim.x; // second blockDim.x entries

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory, handling bounds
    if (idx < N)
    {
        smin[tid] = input[idx];
        smax[tid] = input[idx];
    }
    else
    {
        // Out-of-bounds threads write sentinel values
        smin[tid] = FLT_MAX;     // will not affect min
        smax[tid] = -FLT_MAX;   // will not affect max
    }

    __syncthreads();

    // Parallel reduction for min
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            float val = smin[tid + stride];
            if (val < smin[tid]) smin[tid] = val;
        }
        __syncthreads();
    }

    // Parallel reduction for max
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            float val = smax[tid + stride];
            if (val > smax[tid]) smax[tid] = val;
        }
        __syncthreads();
    }

    // Thread 0 writes block results to global memory
    if (tid == 0)
    {
        blockMin[blockIdx.x] = smin[0];
        blockMax[blockIdx.x] = smax[0];
    }
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_input = (float*)malloc(size);
    float *h_blockMin = NULL;
    float *h_blockMax = NULL;
    float h_min, h_max;

    // Initialize random data
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i)
        h_input[i] = ((float)rand() / RAND_MAX) * 2000.0f - 1000.0f; // range [-1000, 1000]

    // Device pointers
    float *d_input = NULL;
    float *d_blockMin = NULL;
    float *d_blockMax = NULL;

    // Determine block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_input, size));
    CHECK_CUDA(cudaMalloc((void**)&d_blockMin, blocksPerGrid * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_blockMax, blocksPerGrid * sizeof(float)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Allocate host arrays for block results
    h_blockMin = (float*)malloc(blocksPerGrid * sizeof(float));
    h_blockMax = (float*)malloc(blocksPerGrid * sizeof(float));

    // Launch kernel
    size_t sharedMemSize = 2 * threadsPerBlock * sizeof(float); // for min and max
    minMaxKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_blockMin, d_blockMax, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy block results back to host
    CHECK_CUDA(cudaMemcpy(h_blockMin, d_blockMin, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_blockMax, d_blockMax, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

    // Final reduction on host
    h_min = FLT_MAX;
    h_max = -FLT_MAX;
    for (int i = 0; i < blocksPerGrid; ++i)
    {
        if (h_blockMin[i] < h_min) h_min = h_blockMin[i];
        if (h_blockMax[i] > h_max) h_max = h_blockMax[i];
    }

    // Verify correctness with CPU reference
    float cpu_min = h_input[0];
    float cpu_max = h_input[0];
    for (int i = 1; i < N; ++i)
    {
        if (h_input[i] < cpu_min) cpu_min = h_input[i];
        if (h_input[i] > cpu_max) cpu_max = h_input[i];
    }

    printf("GPU min = %f, max = %f\n", h_min, h_max);
    printf("CPU min = %f, max = %f\n", cpu_min, cpu_max);

    // Clean up
    free(h_input);
    free(h_blockMin);
    free(h_blockMax);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_blockMin));
    CHECK_CUDA(cudaFree(d_blockMax));

    return 0;
}
