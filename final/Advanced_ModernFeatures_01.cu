```cpp
/* 
Re-implement vector addition using Unified Memory. Allocate all vectors using `cudaMallocManaged`. 
Notice that you no longer need explicit `cudaMemcpy` calls. The CUDA runtime will migrate data on-demand.

Thinking:
The goal is to replace the usual CUDA pattern of allocating device memory with cudaMalloc, 
copying host data to device with cudaMemcpy, launching the kernel, and copying results back, 
with a simpler Unified Memory approach. By using cudaMallocManaged we allocate a single region 
of memory that is accessible by both the host and the device. The CUDA runtime handles page migration 
between host and device transparently as each side accesses the memory. This eliminates explicit 
memcpy calls, simplifies the code, and demonstrates a new programming model. 
We will still need to synchronize after kernel launch to ensure the device has finished writing 
before the host reads the result. Error checking macros help catch allocation or launch errors. 
The kernel remains the same: each thread adds corresponding elements of two input arrays. 
We will verify correctness by printing a few sample results. */

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    float *A, *B, *C;

    // Allocate Unified Memory – accessible from CPU or GPU
    CUDA_CHECK(cudaMallocManaged(&A, size));
    CUDA_CHECK(cudaMallocManaged(&B, size));
    CUDA_CHECK(cudaMallocManaged(&C, size));

    // Initialize input vectors on the host
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(N - i);
    }

    // Launch kernel with 256 threads per block
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    // Wait for GPU to finish before accessing on host
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify a few results
    printf("Sample results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %f (expected %f)\n", i, C[i], A[i] + B[i]);
    }

    // Free memory
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));

    return 0;
}
```