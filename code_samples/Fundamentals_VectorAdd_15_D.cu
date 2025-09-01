/*
Aim: Implement a kernel that finds the min and max of a vector in a single pass (requires a parallel reduction, which is more advanced).

Thinking:
1. Use a parallel reduction technique: each block processes a chunk of the input, computing local min and max in shared memory.
2. After intra-block reduction, each block writes its local min and max to global arrays blockMins and blockMaxs.
3. Then perform a second reduction over these per-block results to get the final global min and max.
4. The two reductions together give a "single pass" over the data: each element is examined only once by the first kernel.
5. Implemented kernels:
   - minMaxKernel: per-block reduction, writes per-block min/max.
   - reduceMinMaxKernel: reduces the per-block arrays to final min/max.
6. Host code orchestrates memory allocation, data transfer, kernel launches, and final result retrieval.
7. Edge cases: when the number of elements is not a multiple of block size, out-of-range threads are assigned neutral values (INF for min, -INF for max) so they do not affect results.
8. Using fminf and fmaxf for float comparisons; if using integer types, use min/max accordingly.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <float.h>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",\
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Kernel to compute per-block min and max
__global__ void minMaxKernel(const float *input, int n, float *blockMins, float *blockMaxs)
{
    extern __shared__ float shared[];
    float *sMin = shared;
    float *sMax = shared + blockDim.x;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory, handle out-of-bounds
    float val = (idx < n) ? input[idx] : FLT_MAX;
    sMin[tid] = val;
    sMax[tid] = (idx < n) ? val : -FLT_MAX;
    __syncthreads();

    // Intra-block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sMin[tid] = fminf(sMin[tid], sMin[tid + stride]);
            sMax[tid] = fmaxf(sMax[tid], sMax[tid + stride]);
        }
        __syncthreads();
    }

    // Write per-block result to global memory
    if (tid == 0)
    {
        blockMins[blockIdx.x] = sMin[0];
        blockMaxs[blockIdx.x] = sMax[0];
    }
}

// Kernel to reduce per-block min and max arrays to a single min and max
__global__ void reduceMinMaxKernel(const float *inMins, const float *inMaxs, int n,
                                   float *outMin, float *outMax)
{
    extern __shared__ float shared[];
    float *sMin = shared;
    float *sMax = shared + blockDim.x;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    float valMin = (idx < n) ? inMins[idx] : FLT_MAX;
    float valMax = (idx < n) ? inMaxs[idx] : -FLT_MAX;
    sMin[tid] = valMin;
    sMax[tid] = valMax;
    __syncthreads();

    // Reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sMin[tid] = fminf(sMin[tid], sMin[tid + stride]);
            sMax[tid] = fmaxf(sMax[tid], sMax[tid + stride]);
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0)
    {
        outMin[blockIdx.x] = sMin[0];
        outMax[blockIdx.x] = sMax[0];
    }
}

int main(int argc, char *argv[])
{
    // Example vector size
    int n = 1 << 20; // 1M elements
    if (argc > 1)
        n = atoi(argv[1]);

    size_t size = n * sizeof(float);

    // Allocate host memory
    float *h_input = (float *)malloc(size);
    for (int i = 0; i < n; ++i)
        h_input[i] = ((float)rand() / RAND_MAX) * 1000.0f - 500.0f; // Random values [-500, 500]

    // Allocate device memory
    float *d_input;
    CUDA_CHECK(cudaMalloc((void **)&d_input, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Determine grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate per-block min/max arrays
    float *d_blockMins, *d_blockMaxs;
    CUDA_CHECK(cudaMalloc((void **)&d_blockMins, blocksPerGrid * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_blockMaxs, blocksPerGrid * sizeof(float)));

    // Launch first kernel
    size_t sharedMemSize = 2 * threadsPerBlock * sizeof(float);
    minMaxKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, n,
                                                                    d_blockMins,
                                                                    d_blockMaxs);
    CUDA_CHECK(cudaGetLastError());

    // Reduce per-block results to final min and max
    // We will use the same reduction kernel twice if needed
    int currentSize = blocksPerGrid;
    float *d_interMin = d_blockMins;
    float *d_interMax = d_blockMaxs;
    float *d_finalMin = nullptr;
    float *d_finalMax = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_finalMin, sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_finalMax, sizeof(float)));

    while (currentSize > 1)
    {
        int nextBlocks = (currentSize + threadsPerBlock - 1) / threadsPerBlock;
        reduceMinMaxKernel<<<nextBlocks, threadsPerBlock, sharedMemSize>>>(d_interMin,
                                                                          d_interMax,
                                                                          currentSize,
                                                                          d_interMin,
                                                                          d_interMax);
        CUDA_CHECK(cudaGetLastError());
        currentSize = nextBlocks;
    }

    // Copy final results back to host
    float h_min, h_max;
    CUDA_CHECK(cudaMemcpy(&h_min, d_interMin, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_max, d_interMax, sizeof(float), cudaMemcpyDeviceToHost));

    printf("Min: %f\n", h_min);
    printf("Max: %f\n", h_max);

    // Clean up
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_blockMins));
    CUDA_CHECK(cudaFree(d_blockMaxs));
    CUDA_CHECK(cudaFree(d_finalMin));
    CUDA_CHECK(cudaFree(d_finalMax));
    free(h_input);

    return 0;
}
