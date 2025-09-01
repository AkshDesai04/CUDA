/*
Implement in-place multiplication: `A[i] *= B[i]`

The goal of this program is to demonstrate how to perform an element‑wise in‑place multiplication of two arrays on the GPU using CUDA. 
We will allocate two arrays A and B on the host, initialize them with some values, copy them to device memory, launch a kernel that multiplies each element of A by the corresponding element of B, and then copy the modified A array back to the host. Finally, we will print a few elements of the arrays before and after the operation to verify correctness. 

Key aspects of the implementation:

1. **Error checking** – A macro `CUDA_CHECK` is defined to simplify error handling for all CUDA API calls.
2. **Kernel launch configuration** – We calculate the number of blocks required for the given array size and use a typical block size of 256 threads.
3. **In‑place operation** – The kernel directly writes the product back into the `A` array, so no additional output array is needed.
4. **Memory management** – Device memory is allocated and freed properly, and host memory is freed at the end.
5. **Verification** – The program prints the first few elements of the arrays before and after multiplication so that a user can see the change.

This structure serves as a minimal, self‑contained example that can be expanded or integrated into larger CUDA projects.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

// Error checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",     \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),          \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Kernel for in-place multiplication
__global__ void inplace_mul_kernel(float *A, const float *B, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        A[idx] *= B[idx];
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    if (!h_A || !h_B) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays with random values
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)(rand() % 100 + 1);   // avoid zero
        h_B[i] = (float)(rand() % 10 + 1);
    }

    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    inplace_mul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);

    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost));

    // Print a few samples before and after multiplication
    printf("Sample values after in-place multiplication:\n");
    for (int i = 0; i < 10; ++i) {
        printf("A[%d] = %.2f  B[%d] = %.2f  Product = %.2f\n",
               i, h_A[i], i, h_B[i], h_A[i]); // Note: h_A now holds the product
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    free(h_A);
    free(h_B);

    return 0;
}
