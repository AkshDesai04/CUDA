/*
Aim of the program:
Write a kernel that swaps the values between two arrays: `int temp = a[i]; a[i] = b[i]; b[i] = temp;`.

Thinking:
1. The kernel will be a simple element‑wise swap. Each thread handles one element index i.
2. We'll calculate the global thread index with blockIdx, blockDim, and threadIdx.
3. Bounds checking is required to avoid out‑of‑range accesses.
4. Host code will:
   - Allocate and initialize two host arrays.
   - Allocate device memory for each array.
   - Copy host arrays to device.
   - Launch the kernel with enough blocks to cover all elements.
   - Copy results back to host.
   - Print the swapped arrays to verify correctness.
5. Basic error checking after each CUDA call to report failures.
6. The code is written in standard CUDA C, compile with `nvcc` to produce a .cu file.
7. We'll use 256 threads per block as a common choice; the grid size is computed to cover all elements.
8. For simplicity, we use small arrays (e.g., 10 elements) for demonstration, but the code works for any size.
9. The program is fully self‑contained and can be compiled and run on a system with CUDA toolkit installed.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Kernel that swaps corresponding elements of two arrays
__global__ void swapKernel(int *a, int *b, int N) {
    // Compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we don't go out of bounds
    if (idx < N) {
        int temp = a[idx];
        a[idx] = b[idx];
        b[idx] = temp;
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

int main() {
    const int N = 10;  // Number of elements
    int h_a[N], h_b[N];

    // Initialize host arrays with sample data
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;           // 0,1,2,...
        h_b[i] = 100 + i;     // 100,101,102,...
    }

    printf("Before swap:\n");
    for (int i = 0; i < N; ++i) {
        printf("a[%d] = %d, b[%d] = %d\n", i, h_a[i], i, h_b[i]);
    }

    // Allocate device memory
    int *d_a, *d_b;
    CUDA_CHECK(cudaMalloc((void **)&d_a, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_b, N * sizeof(int)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice));

    // Define thread/block configuration
    const int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    swapKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, N);
    CUDA_CHECK(cudaGetLastError());  // Check for launch errors

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b, d_b, N * sizeof(int), cudaMemcpyDeviceToHost));

    printf("\nAfter swap:\n");
    for (int i = 0; i < N; ++i) {
        printf("a[%d] = %d, b[%d] = %d\n", i, h_a[i], i, h_b[i]);
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    return 0;
}
