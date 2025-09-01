/*
Write a separate `__global__` kernel to initialize the input vectors A and B on the GPU, instead of initializing on the host and copying.

Thinking:
- The goal is to move the initialization of the input arrays A and B entirely to device code, eliminating the need for host-side allocation, initialization, and host-to-device memcpy.
- I will create a single global kernel `init_vectors` that writes values into both `A` and `B` based on the thread index. This keeps the code simple and avoids launching multiple kernels.
- Each thread will compute an index `i` and set `A[i] = i` and `B[i] = i * 2` (or any deterministic pattern). Bounds checking will be performed to avoid out-of-range writes.
- After initialization, I will launch the usual vector addition kernel to compute `C = A + B`.
- The program will allocate device memory for A, B, C, launch the initialization kernel, then the addition kernel, copy the result back to host, and print a few elements for verification.
- Error checking macros will be used for clarity.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",                 \
                    __FILE__, __LINE__, err, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

// Kernel to initialize both vectors A and B on the device
__global__ void init_vectors(float *A, float *B, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        A[idx] = (float)idx;          // Example: A[i] = i
        B[idx] = 2.0f * (float)idx;   // Example: B[i] = 2*i
    }
}

// Kernel to perform vector addition C = A + B
__global__ void vector_add(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    float *h_C = (float *)malloc(size);
    if (h_C == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Device pointers
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    // Determine grid dimensions
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel to initialize A and B on the device
    init_vectors<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);
    CUDA_CHECK(cudaGetLastError());

    // Launch vector addition kernel
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Simple verification: print first 10 results
    printf("First 10 results of C = A + B:\n");
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Clean up
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
