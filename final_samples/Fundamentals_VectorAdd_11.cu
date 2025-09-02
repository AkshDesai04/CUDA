/*
Aim of the program: Implement vector addition using a 'grid-stride loop' inside the kernel.
Instead of each thread processing only one element, launch a smaller grid of threads and
have each thread process multiple elements in a strided loop: 
for (int i = thread_id; i < N; i += grid_size) { ... }. 
This makes the kernel independent of the number of threads launched.

Thinking process:
1. Decide on vector size N. For demonstration, set N = 1 << 20 (about one million elements).
2. Use simple float arrays A, B, C. Initialize A[i] = i, B[i] = 2*i.
3. Allocate memory on device using cudaMalloc for each array.
4. Copy host arrays A and B to device.
5. Write a kernel that takes device pointers A, B, C, and N.
   Inside the kernel:
      - Compute global thread index: idx = blockIdx.x * blockDim.x + threadIdx.x.
      - Compute stride = gridDim.x * blockDim.x.
      - Loop over indices i from idx to N in steps of stride, performing C[i] = A[i] + B[i].
6. Launch the kernel with a relatively small number of blocks/threads to demonstrate grid-stride usage.
   For example, blockDim.x = 256, gridDim.x = 64 => stride = 16384. Each thread will process many elements.
7. After kernel execution, copy result C back to host.
8. Verify correctness by printing the first few elements.
9. Add error-checking macro for CUDA calls.
10. Free device memory and exit.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 20)          // Vector size: 1,048,576
#define BLOCK_SIZE 256       // Threads per block
#define GRID_SIZE 64         // Number of blocks (can be much smaller than N/BLOCK_SIZE)

// CUDA error checking macro
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel performing vector addition with grid-stride loop
__global__ void vecAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // Global thread index
    int stride = gridDim.x * blockDim.x;               // Total number of threads

    for (int i = idx; i < n; i += stride) {
        C[i] = A[i] + B[i];
    }
}

int main(void) {
    float *h_A = NULL;
    float *h_B = NULL;
    float *h_C = NULL;
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = 2.0f * (float)i;
    }

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size));

    // Copy input vectors from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel with grid-stride loop
    vecAdd<<<GRID_SIZE, BLOCK_SIZE>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Wait for GPU to finish before accessing on host
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify results (print first 10 elements)
    printf("Verification (first 10 elements):\n");
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %f (expected %f)\n", i, h_C[i], h_A[i] + h_B[i]);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    // Reset device (optional)
    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
