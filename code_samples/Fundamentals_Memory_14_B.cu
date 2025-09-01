```cuda
/*
In a kernel launched with N threads for an array of size N, write to `d_ptr[threadIdx.x + 1]`. Thread N-1 will go out of bounds. Run with `cuda-memcheck`.

My reasoning:
1. The goal is to illustrate a simple out‑of‑bounds memory access that can be detected by cuda‑memcheck.
2. Allocate a device array of N integers (or floats). The size on the device is exactly N.
3. Launch a kernel with N threads in a single block. For each thread i (0 <= i < N) we write
   d_ptr[i + 1] = i + 1;  // This is safe for i from 0 to N-2 but for i = N-1 it writes to index N,
   which is past the end of the array (valid indices 0..N-1).
4. Use a simple error‑checking macro to catch CUDA runtime errors.
5. After the kernel, copy the array back to host and optionally print a few values.
6. The program is intentionally minimal; the key point is that running it under
   `cuda-memcheck ./a.out` will report a "invalid write" at the out‑of‑bounds location.
7. Comments and code are self‑contained in a single .cu file, ready to compile with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for checking CUDA runtime API calls
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Kernel that writes out of bounds for thread N-1
__global__ void out_of_bounds_kernel(int *d_ptr, int N)
{
    int idx = threadIdx.x;
    // Intentionally write one element beyond the allocated array
    d_ptr[idx + 1] = idx + 1;   // For idx = N-1, this writes to d_ptr[N]
}

int main(void)
{
    const int N = 256;                     // Size of the array
    int *d_ptr = NULL;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_ptr, N * sizeof(int)));

    // Initialize device memory to zero (optional)
    CUDA_CHECK(cudaMemset(d_ptr, 0, N * sizeof(int)));

    // Launch kernel with N threads in a single block
    out_of_bounds_kernel<<<1, N>>>(d_ptr, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Wait for device to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy data back to host to inspect
    int h_data[N];
    CUDA_CHECK(cudaMemcpy(h_data, d_ptr, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print first 10 elements (optional)
    printf("First 10 elements after kernel:\n");
    for (int i = 0; i < 10; ++i)
        printf("%d ", h_data[i]);
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_ptr));

    return 0;
}
```