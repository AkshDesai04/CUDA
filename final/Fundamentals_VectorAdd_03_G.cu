```cuda
/*
Aim of the program: Apply the boundary check to a kernel that takes two input vectors and one output vector.

Thinking:
The task is to demonstrate how to protect a CUDA kernel from out‑of‑bounds memory accesses when the number of threads launched may exceed the size of the data vectors. The typical pattern is:
1. Compute a global thread index `idx = blockIdx.x * blockDim.x + threadIdx.x`.
2. If `idx` is less than the vector length, perform the operation (e.g. element‑wise addition).
3. If `idx` is out of range, the thread simply returns without accessing memory.

This kernel will perform element‑wise addition of two input vectors `A` and `B` into an output vector `C`. The host code will:
- Allocate memory on the device for the three vectors.
- Copy input data from host to device.
- Launch the kernel with enough blocks/threads to cover all elements, but rely on the boundary check to avoid invalid accesses.
- Copy the result back to the host.
- Verify the result (simple printout for demonstration).

We also add basic error checking for CUDA API calls to make debugging easier. This example is intentionally simple to keep the focus on the boundary check pattern.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that adds two vectors element‑wise with boundary check
__global__ void vecAdd(const float *A, const float *B, float *C, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",       \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),           \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

int main(void)
{
    const size_t N = 1 << 20;   // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    // Initialize input data
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("Sample results:\n");
    for (int i = 0; i < 5; ++i) {
        printf("C[%d] = %f (expected %f)\n", i, h_C[i], h_A[i] + h_B[i]);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```