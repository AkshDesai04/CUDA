```cpp
/*
Aim of the program: 
Write a separate __global__ kernel to initialize the input vectors A and B on the GPU, instead of initializing on the host and copying.

Thinking:
The original approach often initializes input arrays on the host, copies them to the device, performs computation, and copies results back. To eliminate the host-to-device copy for the input data, we can allocate device memory and then launch a CUDA kernel that writes the initial values directly into the device arrays. This reduces data movement overhead and can improve performance, especially for large vectors. 

We will:
1. Define a __global__ kernel `initVectors` that takes device pointers for A and B, the number of elements N, and initializes each element based on its index (e.g., A[i] = i, B[i] = 2*i).
2. Define another __global__ kernel `addVectors` that performs element-wise addition C[i] = A[i] + B[i].
3. In `main`, allocate device memory for A, B, and C, launch `initVectors` to populate A and B, then launch `addVectors`, copy the result back to host, and verify or print a few elements.
4. Include error checking macros for CUDA API calls.
*/

#include <cstdio>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel to initialize vectors A and B on the device
__global__ void initVectors(float *A, float *B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        A[idx] = static_cast<float>(idx);        // Example: A[i] = i
        B[idx] = static_cast<float>(2 * idx);    // Example: B[i] = 2*i
    }
}

// Kernel to perform vector addition C = A + B
__global__ void addVectors(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const int N = 1 << 20;  // Number of elements (1M)
    const size_t bytes = N * sizeof(float);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    // Define execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel to initialize A and B on device
    initVectors<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);
    CUDA_CHECK(cudaGetLastError());

    // Launch kernel to add vectors
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Allocate host memory for result
    float *h_C = (float *)malloc(bytes);
    if (h_C == nullptr) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Simple verification: print first 10 results
    printf("First 10 results of C = A + B:\n");
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Cleanup
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```