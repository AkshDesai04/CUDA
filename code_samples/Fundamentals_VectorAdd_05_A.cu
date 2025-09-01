```cuda
/*
Aim of the program:
Implement C[i] = A[i] * A[i] (squaring).

Thinking process:
1. The program will be written in CUDA C and compiled with nvcc.
2. We'll create a simple element‑wise squaring kernel that takes an input array A and writes A[i]*A[i] to C.
3. To keep the example self‑contained, the host code will:
   - Define a constant array size N (e.g., 1<<20 elements).
   - Allocate host memory for arrays A and C.
   - Initialize A with some values (e.g., sequential integers converted to float).
   - Allocate device memory for A and C.
   - Copy A from host to device.
   - Determine a suitable grid and block size for launching the kernel.
   - Launch the kernel to compute C on the device.
   - Copy the result C back to host.
   - Print a few elements of C to verify correctness.
   - Clean up device and host memory.
4. Error checking after each CUDA API call will be added for robustness.
5. The code uses standard C++ headers and CUDA runtime API functions.
6. The kernel uses the standard "if (i < N)" guard to handle arrays whose size is not a multiple of the block size.

The program is straightforward, focusing on clarity rather than performance optimizations.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N (1 << 20)            // Number of elements (1,048,576)
#define THREADS_PER_BLOCK 256  // Common block size

// CUDA kernel: compute C[i] = A[i] * A[i]
__global__ void squareKernel(const float *A, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = A[idx];
        C[idx] = val * val;
    }
}

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",      \
                    __func__, __FILE__, __LINE__,                 \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

int main() {
    // Host arrays
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));
    if (!h_A || !h_C) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host array A with some values
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
    }

    // Device arrays
    float *d_A = nullptr;
    float *d_C = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, N * sizeof(float)));

    // Copy input data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));

    // Determine grid dimensions
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch the kernel
    squareKernel<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_C, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("First 10 results of C[i] = A[i]^2:\n");
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_C);

    return 0;
}
```