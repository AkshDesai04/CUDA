```cpp
/*
The second kernel is launched with just a single block. This block loads the d_partial_sums array into shared memory and performs a final reduction on it.

I want to write a CUDA program that demonstrates a two‑step reduction: the first kernel computes partial sums per block, and the second kernel collapses those partial sums into a single final sum. The specification says that the second kernel should be launched with a single block, which will load the array of partial sums (`d_partial_sums`) into shared memory and perform a reduction.  
To satisfy that I:
1. Pick a problem size that fits in a single block for the final kernel (so the number of partial sums must be <= the maximum number of threads per block, typically 1024).  
2. Implement the first kernel (`reduceKernel`) with a classic two‑thread‑per‑element pattern, storing intermediate sums in shared memory and reducing them to a single value per block.  
3. Store each block’s result in `d_partial_sums`.  
4. Implement the second kernel (`finalReduceKernel`) with one block. The kernel will copy the partial sums into shared memory (`sdata`), then perform the same in‑block reduction.  
5. In `main` I allocate host and device memory, initialize the input array with a simple value (1.0f), launch the two kernels with appropriate launch parameters, copy back the final result, and print it.  
6. For simplicity and correctness I use a static input size of 262144 (2^18) which, with a block size of 256 and two elements per thread, produces 512 partial sums – well within the 1024‑thread limit of a single block.  
7. I also include a small `cudaCheckError` macro for debugging.  
All the code is contained in a single .cu file, with a top‑level comment containing the exact wording of the aim and the reasoning above, followed by the full CUDA C program. 
*/

#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>

// Simple CUDA error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// First kernel: compute partial sums per block
__global__ void reduceKernel(const float *d_in, float *d_partial, int N)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + tid;
    float mySum = 0.0f;
    if (i < N) mySum = d_in[i];
    if (i + blockDim.x < N) mySum += d_in[i + blockDim.x];
    sdata[tid] = mySum;
    __syncthreads();

    // In‑block reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // Write result for this block to global memory
    if (tid == 0) d_partial[blockIdx.x] = sdata[0];
}

// Second kernel: final reduction, launched with a single block
__global__ void finalReduceKernel(const float *d_partial, float *d_out, int numPartial)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    // Load partial sums into shared memory
    if (tid < numPartial)
        sdata[tid] = d_partial[tid];
    else
        sdata[tid] = 0.0f;
    __syncthreads();

    // In‑block reduction
    for (unsigned int s = numPartial / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // The first thread writes the final sum
    if (tid == 0) d_out[0] = sdata[0];
}

int main()
{
    // Problem size: 2^18 elements
    const int N = 1 << 18;              // 262,144 elements
    const int blockSize = 256;          // Threads per block
    const int threadsPerElement = 2;    // Two elements per thread in first kernel
    const int numBlocks = (N + blockSize * threadsPerElement - 1) / (blockSize * threadsPerElement);

    printf("N = %d, blockSize = %d, numBlocks = %d\n", N, blockSize, numBlocks);

    // Host allocation and initialization
    float *h_in = new float[N];
    for (int i = 0; i < N; ++i)
        h_in[i] = 1.0f;  // Simple value so sum = N

    // Device allocations
    float *d_in = nullptr;
    float *d_partial = nullptr;
    float *d_out = nullptr;
    cudaCheckError(cudaMalloc((void**)&d_in, N * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_partial, numBlocks * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_out, sizeof(float)));

    // Copy input to device
    cudaCheckError(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch first kernel
    size_t sharedMemSize = blockSize * sizeof(float);
    reduceKernel<<<numBlocks, blockSize, sharedMemSize>>>(d_in, d_partial, N);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // Launch second kernel with a single block
    // We need enough shared memory for all partial sums
    sharedMemSize = numBlocks * sizeof(float);
    finalReduceKernel<<<1, numBlocks, sharedMemSize>>>(d_partial, d_out, numBlocks);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // Retrieve result
    float h_out = 0.0f;
    cudaCheckError(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    printf("Computed sum = %f\n", h_out);
    printf("Expected sum = %f\n", (float)N);

    // Clean up
    delete[] h_in;
    cudaCheckError(cudaFree(d_in));
    cudaCheckError(cudaFree(d_partial));
    cudaCheckError(cudaFree(d_out));

    return 0;
}
```