```cuda
/*
Aim of the program:
Implement vector-scalar addition: B[i] = A[i] + 5; where 5 is a scalar value passed to the kernel.

Thinking:
The goal is to demonstrate a simple element‑wise operation on a vector where each element
is incremented by a constant scalar.  The scalar value is passed as an argument to the
CUDA kernel, allowing the host code to specify different values without changing the
kernel source.  The implementation follows these steps:

1. **Define constants** – The vector size `N` is set to 1024 for demonstration purposes,
   but the code can be adapted to larger sizes.  The number of threads per block and
   blocks per grid are computed so that all elements are processed.

2. **Memory allocation** – Host memory is allocated with `malloc` for the input vector
   `h_A` and output vector `h_B`.  Device memory is allocated using `cudaMalloc` for
   `d_A` and `d_B`.

3. **Initialization** – The input vector `h_A` is filled with sequential values
   (0.0f, 1.0f, …).  This makes it easy to verify the addition result.

4. **Kernel definition** – `vecAddScalar` is a straightforward kernel that calculates
   a global index `idx`.  If the index is within bounds, it writes `B[idx] = A[idx] + scalar`.

5. **Kernel launch** – The kernel is launched with the computed grid and block
   dimensions.  The scalar `5.0f` is passed as an argument.  A device synchronization
   (`cudaDeviceSynchronize`) guarantees completion before copying data back.

6. **Result retrieval** – The output vector is copied back to host memory with
   `cudaMemcpy`.  The first few elements are printed to confirm correctness.

7. **Cleanup** – All allocated memory on both host and device is freed.

8. **Error handling** – A simple macro `CUDA_CHECK` wraps CUDA API calls to catch
   and report errors immediately.  This keeps the main logic clean while ensuring
   robustness.

The program is self‑contained, can be compiled with `nvcc`, and will produce
output that verifies the vector‑scalar addition.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define N 1024                     // Size of the vectors
#define THREADS_PER_BLOCK 256      // Common choice for many GPUs

// Simple error checking macro
#define CUDA_CHECK(call)                                 \
    do {                                                \
        cudaError_t err = call;                         \
        if (err != cudaSuccess) {                       \
            fprintf(stderr, "CUDA error in %s:%d: %s\n",\
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                         \
        }                                               \
    } while (0)

// Kernel: B[i] = A[i] + scalar
__global__ void vecAddScalar(const float *A, float *B, float scalar, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        B[idx] = A[idx] + scalar;
    }
}

int main(void)
{
    // Allocate host memory
    float *h_A = (float *)malloc(N * sizeof(float));
    float *h_B = (float *)malloc(N * sizeof(float));
    if (!h_A || !h_B)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input vector
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;
    }

    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_B, N * sizeof(float)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));

    // Compute grid dimensions
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    float scalar = 5.0f; // The scalar to add

    // Launch kernel
    vecAddScalar<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_A, d_B, scalar, N);
    CUDA_CHECK(cudaGetLastError());      // Check launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure kernel finished

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_B, d_B, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print first 10 results for verification
    printf("First 10 results of B = A + %.2f:\n", scalar);
    for (int i = 0; i < 10 && i < N; ++i)
    {
        printf("B[%d] = %.2f (A[%d] = %.2f + %.2f)\n",
               i, h_B[i], i, h_A[i], scalar);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    free(h_A);
    free(h_B);

    return EXIT_SUCCESS;
}
```