/*
Implement a clamp function: C[i] = min(max(A[i], min_val), max_val).

The aim of this program is to demonstrate a simple CUDA kernel that applies a clamp operation
to each element of an input array A.  For each index i, we compute the clamped value
by first taking the maximum of A[i] and a lower bound min_val, and then taking the minimum
of that intermediate result and an upper bound max_val.  The final value is stored in array C.

The program contains:
- A __global__ kernel clampKernel that performs the clamp per thread.
- Host code that allocates input and output arrays, initializes input with sample data,
  copies data to the device, launches the kernel, copies the result back, and prints it.
- Basic error checking for CUDA API calls.
- Use of fmaxf and fminf for floating-point clamping.
- Simple test case that includes values below min_val, within range, and above max_val.

The kernel uses a 1D grid of 1D blocks for simplicity.  Each thread processes one element
if its global index is within bounds.  The launch configuration is chosen to cover all
elements with a block size of 256 threads per block.

Compile with:
    nvcc -o clamp clamp.cu

Run with:
    ./clamp
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel that clamps each element of input array A into [min_val, max_val]
__global__ void clampKernel(const float *A, float *C, int n, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = A[idx];
        // Clamp: first max with min_val, then min with max_val
        float clamped = fminf(fmaxf(val, min_val), max_val);
        C[idx] = clamped;
    }
}

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                                          \
    do {                                                                           \
        cudaError_t err = call;                                                    \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                                                   \
        }                                                                          \
    } while (0)

int main(void) {
    const int n = 10;                 // Number of elements
    const float min_val = 0.0f;       // Minimum clamp value
    const float max_val = 10.0f;      // Maximum clamp value

    // Host arrays
    float h_A[n];
    float h_C[n];

    // Initialize input with a mix of values
    for (int i = 0; i < n; ++i) {
        h_A[i] = (float)(i * 2 - 5);   // Will produce values: -5, -3, -1, 1, 3, 5, 7, 9, 11, 13
    }

    // Device pointers
    float *d_A = NULL;
    float *d_C = NULL;

    size_t size_bytes = n * sizeof(float);

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_A, size_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size_bytes));

    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    clampKernel<<<blocks, threads_per_block>>>(d_A, d_C, n, min_val, max_val);

    // Check for launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_bytes, cudaMemcpyDeviceToHost));

    // Print results
    printf("Input\t\tClamp Output\n");
    for (int i = 0; i < n; ++i) {
        printf("%6.2f\t\t%6.2f\n", h_A[i], h_C[i]);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
