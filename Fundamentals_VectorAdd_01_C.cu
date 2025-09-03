```cpp
/*
Write a separate `__global__` kernel to initialize the input vectors A and B on the GPU, 
instead of initializing on the host and copying.

Thinking:

The requirement is to move the initialization of the input data from the host side to 
the device side. This involves writing a __global__ kernel that writes initial values 
into two arrays A and B that reside on the GPU. After allocation of device memory, 
the host will launch this kernel to set the values. Following this, we can perform 
operations such as vector addition using another kernel. The main steps are:

1. Define the size of the vectors (N).
2. Allocate device memory for A, B, and C.
3. Implement an init kernel that writes a known value into each element of A and B.
4. Launch the init kernel with enough threads to cover all elements.
5. (Optional) Implement a vector addition kernel to demonstrate usage.
6. Launch the vector addition kernel.
7. Copy the result back to host and verify.
8. Free device memory.

Error checking is included to catch any CUDA runtime errors. 
The program is a minimal, self‑contained example that can be compiled with nvcc.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 20) // 1M elements

// Kernel to initialize two vectors A and B on the device
__global__ void initVectors(float *A, float *B, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        A[idx] = 2.0f; // arbitrary value
        B[idx] = 3.0f; // arbitrary value
    }
}

// Simple vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        C[idx] = A[idx] + B[idx];
    }
}

// Utility macro for error checking
#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

int main(void)
{
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    float *h_C = NULL;
    size_t size = N * sizeof(float);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    // Launch kernel to initialize A and B
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    initVectors<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch vector addition kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Allocate host memory for result
    h_C = (float *)malloc(size);
    if (!h_C)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("Sample results:\n");
    for (int i = 0; i < 5; ++i)
        printf("C[%d] = %f\n", i, h_C[i]);

    // Clean up
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    printf("Program completed successfully.\n");
    return EXIT_SUCCESS;
}
```