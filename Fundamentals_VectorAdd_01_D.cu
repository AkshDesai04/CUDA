/*
Perform the vector addition in-place: `A[i] = A[i] + B[i]`.

Thinking:
- The goal is to add two vectors element‑wise and write the result back into the first vector (in‑place).
- We'll write a simple CUDA kernel that each thread handles one element: `A[idx] += B[idx]` if the index is within bounds.
- In the host code we will:
  1. Allocate host memory for vectors A and B, and initialize them with some test data.
  2. Allocate device memory for the same vectors.
  3. Copy the host data to device.
  4. Launch the kernel with enough blocks and threads to cover all elements.
  5. Copy the result back to the host.
  6. Verify a few elements and clean up.
- Error checking will be added after each CUDA API call to ensure correct execution.
- We'll use a simple vector size (e.g., 1 million elements) and float data type for demonstration.
- The program is self‑contained and can be compiled with `nvcc -o vecAddInPlace vecAddInPlace.cu` and run directly.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel for in-place vector addition
__global__ void vecAddInPlace(float *A, const float *B, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        A[idx] += B[idx];
    }
}

int main(void)
{
    const int N = 1 << 20;          // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    if (!h_A || !h_B) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;            // e.g., 0, 1, 2, ...
        h_B[i] = 1.0f;                // add 1 to each element
    }

    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vecAddInPlace<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);
    CHECK_CUDA(cudaGetLastError());

    // Synchronize
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("Sample results after in-place addition:\n");
    for (int i = 0; i < 5; ++i) {
        printf("h_A[%d] = %f\n", i, h_A[i]);
    }

    // Clean up
    free(h_A);
    free(h_B);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));

    return EXIT_SUCCESS;
}
