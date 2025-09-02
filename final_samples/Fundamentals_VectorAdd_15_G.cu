```cuda
/*
Fuse a copy and a scale operation: `B[i] = A[i] * scalar`.

The goal of this program is to demonstrate how a simple fused operation, namely scaling
an input array by a scalar and writing the result to an output array, can be performed
directly on the GPU using a single CUDA kernel. This eliminates the need for separate
copy and multiply passes, reducing kernel launch overhead and memory traffic.

Thinking process:
- The operation is element‑wise, so each thread can process one element of the arrays.
- A straightforward kernel takes three arguments: the input array `A`, the output array
  `B`, and the scaling factor `scalar`. It computes the index based on block and thread
  indices and performs the multiplication if the index is within bounds.
- In `main` we:
  1. Define array size and allocate host memory for input and output.
  2. Initialize the input array with some values.
  3. Allocate device memory for `A` and `B` and copy the input to the device.
  4. Launch the kernel with a configuration that covers all elements.
  5. Copy the result back to the host and verify it.
  6. Clean up all allocated memory.
- Error checking macros (`CUDA_CHECK`) are used to catch any runtime issues.
- The program prints the first few results to confirm correctness.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple error checking macro
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Kernel that performs B[i] = A[i] * scalar
__global__ void fuse_copy_scale(const float *A, float *B, float scalar, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        B[idx] = A[idx] * scalar;
    }
}

int main(void)
{
    const size_t N = 1 << 20;          // 1M elements
    const size_t bytes = N * sizeof(float);
    const float scalar = 3.14f;

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    if (!h_A || !h_B) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input array
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i) * 0.01f;  // example values
    }

    // Allocate device memory
    float *d_A = nullptr;
    float *d_B = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    fuse_copy_scale<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, scalar, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("First 10 results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_B[%d] = %f (expected %f)\n", i, h_B[i], h_A[i] * scalar);
    }

    // Clean up
    free(h_A);
    free(h_B);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    return EXIT_SUCCESS;
}
```