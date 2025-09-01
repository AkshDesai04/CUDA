/*
Aim: Explain why calling this check function only at the end of the program is not as useful for debugging.

Thinking:
- In CUDA, kernel launches are asynchronous. The host returns immediately after launching a kernel,
  and any errors (e.g., invalid configuration, out‑of‑bounds memory access) are only recorded
  in the device error state.
- cudaGetLastError() clears that error state, so if you call it only once at the end of the
  program you only learn that *some* error happened, but you lose the context of *where*
  and *why* it happened. You cannot determine which kernel launch or memory copy
  caused the failure, especially if multiple launches precede the final check.
- Debugging is more effective when the error state is inspected immediately after each
  CUDA API call or kernel launch. That way you can pinpoint the exact location
  (e.g., a particular kernel or launch configuration) that triggers the fault.
- Additionally, CUDA launches are queued: an earlier failure can prevent later
  kernels from executing at all. If you only check at the end, the program might
  have already finished many kernels that never ran or produced incorrect results,
  masking the true cause of the failure.
- The following program demonstrates this by intentionally launching a kernel that
  writes out of bounds. It shows the difference between checking after each launch
  and checking only once at the end, illustrating why the latter is less useful for
  debugging.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro to check CUDA errors
#define CHECK_CUDA(call)                                                    \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Kernel that deliberately writes out of bounds
__global__ void out_of_bounds_kernel(int *data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        data[idx] = idx; // If N > array size on device, this writes out of bounds
}

// Simple kernel that adds two arrays
__global__ void vector_add(const int *a, const int *b, int *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}

int main(void)
{
    const int N = 20;          // Logical array size for kernels
    const int devN = 10;       // Actual size allocated on device
    const size_t bytes = N * sizeof(int);

    // Allocate host memory
    int *h_a = (int*)malloc(bytes);
    int *h_b = (int*)malloc(bytes);
    int *h_c = (int*)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // Allocate device memory (only devN elements)
    int *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void**)&d_a, devN * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_b, devN * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_c, devN * sizeof(int)));

    // Copy data from host to device (only devN elements)
    CHECK_CUDA(cudaMemcpy(d_a, h_a, devN * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, devN * sizeof(int), cudaMemcpyHostToDevice));

    // Launch vector_add correctly
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // **Option 1: Check after each launch (recommended)**
    // Uncomment the following line to see the immediate error detection
    // CHECK_CUDA(cudaGetLastError());

    // Launch out_of_bounds_kernel intentionally (this will cause an error)
    out_of_bounds_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_c, N);

    // **Option 2: Check only at the end (illustrated below)**
    // Comment out the following line to skip the final check
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, devN * sizeof(int), cudaMemcpyDeviceToHost));

    // Print first few results
    printf("First 5 results:\n");
    for (int i = 0; i < 5; ++i)
        printf("%d ", h_c[i]);
    printf("\n");

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
