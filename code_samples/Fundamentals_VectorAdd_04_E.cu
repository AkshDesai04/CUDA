/*
Implement in-place subtraction: `A[i] -= B[i]`.

The goal of this program is to demonstrate a simple CUDA kernel that performs an element‑wise subtraction of two arrays, updating the first array in place. The steps involved are:

1. **Data preparation on the host** – Create two input arrays `A` and `B` of a given size `N`, and optionally initialize them with sample values.
2. **Memory allocation on the device** – Allocate GPU memory for `A_d` and `B_d`.
3. **Copy data from host to device** – Transfer `A` and `B` to the GPU.
4. **Launch the kernel** – Each thread will compute a global index and perform `A_d[idx] -= B_d[idx]`. The kernel is launched with enough blocks to cover all `N` elements.
5. **Copy results back to host** – Transfer the updated `A_d` back to `A` on the CPU.
6. **Cleanup** – Free GPU and CPU memory.
7. **Optional validation** – Print a few elements to verify the operation.

The kernel is intentionally straightforward, as the focus is on demonstrating in‑place subtraction rather than performance optimization. Error checking is performed after each CUDA API call using a helper macro `CUDA_CHECK`. This ensures that any failures are caught early and reported with context. The program compiles as a CUDA C file (`.cu`) and can be built with `nvcc`.

*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",     \
                    #call, __FILE__, __LINE__,                   \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Kernel that subtracts B from A in-place: A[i] -= B[i]
__global__ void subtractKernel(float *A, const float *B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        A[idx] -= B[idx];
    }
}

int main(void) {
    const int N = 1 << 20;  // Example size: 1M elements
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    if (!h_A || !h_B) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;          // Example: 0, 1, 2, ...
        h_B[i] = (float)(N - i);    // Example: N, N-1, N-2, ...
    }

    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Determine kernel launch configuration
    const int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    subtractKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost));

    // Simple verification: print first 10 results
    printf("First 10 results of A[i] -= B[i]:\n");
    for (int i = 0; i < 10; ++i) {
        printf("A[%d] = %f\n", i, h_A[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    free(h_A);
    free(h_B);

    return 0;
}
