```cuda
/*
 * Aim of the program (passed in this prompt word to word):
 * Implement a version where the first kernel uses a grid-stride loop, so each thread processes multiple input elements before the block-level reduction begins.
 *
 * My thinking while writing this code:
 * 1. The requirement is to demonstrate a grid‑stride loop in the first kernel.  
 *    That kernel will accumulate a local sum for each thread by stepping through
 *    the input array in strides equal to the total number of threads in the grid.
 * 2. After the grid‑stride accumulation, we perform a standard block‑level
 *    reduction using shared memory to obtain a partial sum per block.
 * 3. Those partial sums are written to a temporary array.  A second kernel
 *    (finalReduceKernel) reduces this array to a single value.  The second
 *    kernel can use a simple block‑level reduction; it does not need a
 *    grid‑stride loop because the number of elements is already small.
 * 4. On the host we repeatedly launch finalReduceKernel until only one
 *    element remains.  Finally we copy the result back to the host and print
 *    it.
 * 5. For demonstration, we allocate a large array of floats, fill it with
 *    random values, perform the reduction, and compare the CUDA result with
 *    a CPU reference to verify correctness.
 * 6. The code uses minimal error checking and is fully self‑contained.
 * 7. The grid‑stride logic is clearly visible in the first kernel.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Utility macro for error checking
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",             \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

// First kernel: grid‑stride accumulation + block reduction
__global__ void reduceKernel(const float *input, float *partial, size_t N)
{
    extern __shared__ float sdata[];

    unsigned int tid   = threadIdx.x;
    unsigned int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride= gridDim.x * blockDim.x;

    float sum = 0.0f;
    // Grid‑stride loop: each thread processes many elements
    for (unsigned int i = idx; i < N; i += stride)
    {
        sum += input[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Block reduction using shared memory
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0)
        partial[blockIdx.x] = sdata[0];
}

// Second kernel: final reduction of partial sums (no grid‑stride needed)
__global__ void finalReduceKernel(float *partial, size_t N)
{
    extern __shared__ float sdata[];

    unsigned int tid   = threadIdx.x;
    unsigned int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride= gridDim.x * blockDim.x;

    float sum = 0.0f;
    // Use grid‑stride here only if the number of partials is still large;
    // it is safe but not required.
    for (unsigned int i = idx; i < N; i += stride)
    {
        sum += partial[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Block reduction
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        partial[blockIdx.x] = sdata[0];
}

// CPU reference reduction for verification
float cpuReduce(const float *data, size_t N)
{
    float sum = 0.0f;
    for (size_t i = 0; i < N; ++i)
        sum += data[i];
    return sum;
}

int main(int argc, char *argv[])
{
    // Size of input array
    size_t N = 1 << 20; // 1M elements
    if (argc > 1)
        N = atol(argv[1]);

    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_input = (float *)malloc(bytes);
    if (!h_input) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input with random values
    for (size_t i = 0; i < N; ++i)
        h_input[i] = (float)(rand()) / RAND_MAX;

    // Allocate device memory
    float *d_input = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_input, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Determine block and grid sizes
    const unsigned int blockSize = 256;
    unsigned int numBlocks = (N + blockSize - 1) / blockSize;

    // Allocate array for partial sums
    float *d_partial = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_partial, numBlocks * sizeof(float)));

    // Launch first kernel
    size_t sharedMemSize = blockSize * sizeof(float);
    reduceKernel<<<numBlocks, blockSize, sharedMemSize>>>(d_input, d_partial, N);
    CUDA_CHECK(cudaGetLastError());

    // Reduce partial sums to single value
    size_t num = numBlocks;
    while (num > 1)
    {
        unsigned int gsize = (num + blockSize - 1) / blockSize;
        finalReduceKernel<<<gsize, blockSize, sharedMemSize>>>(d_partial, num);
        CUDA_CHECK(cudaGetLastError());
        num = gsize;
    }

    // Copy result back to host
    float gpu_result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpu_result, d_partial, sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference
    float cpu_result = cpuReduce(h_input, N);

    printf("CUDA reduction result: %f\n", gpu_result);
    printf("CPU reference result:  %f\n", cpu_result);
    printf("Difference:            %f\n", fabsf(gpu_result - cpu_result));

    // Clean up
    free(h_input);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_partial));

    return 0;
}
```