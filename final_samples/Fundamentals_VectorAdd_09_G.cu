/*
Aim of the program:
Implement `D[i] = (A[i] * s1) + (B[i] * s2)`.

Thinking:
To create a simple CUDA program that demonstrates basic device memory management,
kernel launch, and element‑wise arithmetic, the program follows these steps:

1. **Define problem size** – Choose a convenient size (e.g., N = 1<<20) and
   allocate host arrays `h_A`, `h_B`, and `h_D`.
2. **Initialize host data** – Fill `h_A` and `h_B` with sample values (e.g.,
   `A[i] = i`, `B[i] = 2*i`) and set scalar coefficients `s1` and `s2`.
3. **Allocate device memory** – Allocate `d_A`, `d_B`, and `d_D` on the GPU.
4. **Copy inputs to device** – Use `cudaMemcpy` to transfer `h_A` and `h_B`
   to `d_A` and `d_B`.
5. **Launch kernel** – Compute each element of `D` in parallel:
   `D[i] = A[i] * s1 + B[i] * s2`.  
   Use a 1‑D grid of 1‑D blocks, with a block size of 256 threads.
6. **Copy result back to host** – Transfer `d_D` back to `h_D`.
7. **Verify and clean up** – Optionally print a few values to confirm correctness
   and free all allocated memory.

The code includes basic error checking via a macro to make debugging easier.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N (1<<20)          // Number of elements (1,048,576)
#define BLOCK_SIZE 256     // Threads per block

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel to compute D[i] = A[i] * s1 + B[i] * s2
__global__ void computeKernel(const float *A, const float *B, float *D,
                              float s1, float s2, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        D[idx] = A[idx] * s1 + B[idx] * s2;
    }
}

int main(void)
{
    // Host pointers
    float *h_A = NULL;
    float *h_B = NULL;
    float *h_D = NULL;

    // Device pointers
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_D = NULL;

    size_t bytes = N * sizeof(float);

    // Allocate host memory
    h_A = (float *)malloc(bytes);
    h_B = (float *)malloc(bytes);
    h_D = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_D) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host data
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = (float)i;        // Example values
        h_B[i] = (float)(2 * i);
    }

    float s1 = 3.5f;  // Example scalar coefficients
    float s2 = 1.2f;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_D, bytes));

    // Copy inputs from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Determine grid dimensions
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    computeKernel<<<gridDim, blockDim>>>(d_A, d_B, d_D, s1, s2, N);
    CUDA_CHECK(cudaGetLastError());   // Check kernel launch

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost));

    // Optional: Verify a few elements
    printf("Sample output (index : value):\n");
    for (size_t i = 0; i < 10; ++i) {
        printf("D[%zu] = %f\n", i, h_D[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_D));
    free(h_A);
    free(h_B);
    free(h_D);

    // Reset device
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
