```cpp
/*
Aim: Combine the approaches: a single kernel computes block-level partial sums, and then thread 0 of each block uses atomicAdd on a single global counter (*d_final_sum) to add its result.

Thinking:
- The kernel will perform a parallel reduction within each block using shared memory to compute a block‑level sum.
- Each thread will process multiple elements of the input array by striding over it, accumulating a local sum.
- After loading local sums into shared memory, a tree‑style reduction is performed to compute the final block sum.
- Once a block sum is ready, only thread 0 will add it to the global sum using atomicAdd to avoid race conditions.
- This approach keeps the global memory traffic low (only one atomic add per block) and keeps the reduction on the GPU.
- In the host code we allocate memory, initialize data, copy to device, launch the kernel, copy back the final sum, and clean up.
- We also add simple CUDA error checking to aid debugging.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Simple CUDA error checking macro
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

// Kernel that sums an array using block partial sums and atomicAdd
__global__ void sumKernel(const float *d_in, float *d_final_sum, size_t N)
{
    extern __shared__ float sdata[];

    // Each thread loads elements into local sum
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float localSum = 0.0f;

    // Process two elements per thread to reduce global memory accesses
    if (idx < N) localSum += d_in[idx];
    if (idx + blockDim.x < N) localSum += d_in[idx + blockDim.x];

    // Store local sum into shared memory
    sdata[tid] = localSum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes block sum to global memory via atomicAdd
    if (tid == 0) {
        atomicAdd(d_final_sum, sdata[0]);
    }
}

int main(void)
{
    const size_t N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_in = (float*)malloc(bytes);
    if (!h_in) { fprintf(stderr, "Failed to allocate host memory\n"); return EXIT_FAILURE; }

    // Initialize data
    for (size_t i = 0; i < N; ++i) {
        h_in[i] = 1.0f; // simple test: sum should be N
    }

    // Allocate device memory
    float *d_in = nullptr;
    float *d_final_sum = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_final_sum, sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Initialize global sum to zero
    CUDA_CHECK(cudaMemset(d_final_sum, 0, sizeof(float)));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
    size_t sharedMemSize = threadsPerBlock * sizeof(float);
    sumKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_in, d_final_sum, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back
    float h_final_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_final_sum, d_final_sum, sizeof(float), cudaMemcpyDeviceToHost));

    // Verify result
    printf("Computed sum = %f\n", h_final_sum);
    printf("Expected sum = %f\n", (float)N);

    // Clean up
    cudaFree(d_in);
    cudaFree(d_final_sum);
    free(h_in);

    return 0;
}
```