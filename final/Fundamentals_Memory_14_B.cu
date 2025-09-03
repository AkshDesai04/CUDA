```cpp
/*
 * Aim of the program: In a kernel launched with N threads for an array of size N, write to d_ptr[threadIdx.x + 1]. Thread N-1 will go out of bounds. Run with cuda-memcheck.
 *
 * Thinking:
 * 1. We want to illustrate a simple out-of-bounds write by launching N threads on an array of size N and accessing element idx+1.
 * 2. The kernel will write to d_ptr[idx+1] unconditionally for idx < N, so the last thread (idx = N-1) will write to d_ptr[N] which is one element beyond the allocated memory.
 * 3. The host code will allocate the array, launch the kernel with exactly N threads in a single block, synchronize, copy the data back, and print it.
 * 4. When run under `cuda-memcheck`, it should report a device error (invalid write) for the out-of-bounds access.
 * 5. The code includes basic error checking for CUDA API calls to aid debugging.
 */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void writeOutOfBounds(int *d_ptr, int N)
{
    int idx = threadIdx.x;
    // No bounds check on the write to intentionally create an OOB write
    d_ptr[idx + 1] = idx;
}

void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s : %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    const int N = 10;  // Size of the array
    int *d_ptr = nullptr;
    int h[N];

    // Allocate device memory
    checkCuda(cudaMalloc((void**)&d_ptr, N * sizeof(int)), "cudaMalloc");

    // Initialize device memory to zero
    checkCuda(cudaMemset(d_ptr, 0, N * sizeof(int)), "cudaMemset");

    // Launch kernel with N threads in a single block
    writeOutOfBounds<<<1, N>>>(d_ptr, N);

    // Synchronize and check for kernel launch errors
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after kernel");

    // Copy result back to host
    checkCuda(cudaMemcpy(h, d_ptr, N * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy HtoD");

    // Print the host array
    printf("Host array after kernel launch:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%d ", h[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_ptr);
    return 0;
}
```