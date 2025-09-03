/*
Run the version without error checking under `cuda-memcheck`. Observe that `cuda-memcheck` finds the error even if the program itself doesn't report one.

This program demonstrates a deliberate out-of-bounds memory access on the device. The kernel writes to an array index that is beyond the allocated device memory size. The host code does not perform any error checking after CUDA API calls or kernel launches. As a result, the program will run and print some output, but it will not report any error. When the executable is run under `cuda-memcheck`, the tool detects the out-of-bounds write and reports it.

The idea is to illustrate how `cuda-memcheck` can uncover hidden bugs that are not caught by basic error handling in the host code.
*/

#include <stdio.h>
#include <stdlib.h>

__global__ void out_of_bounds_kernel(int *d_arr, int size)
{
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Intentionally write out of bounds
    // If idx >= size, this is an illegal memory write
    d_arr[idx] = idx;
}

int main()
{
    const int N = 32;                 // Size of array on device
    const int THREADS_PER_BLOCK = 64; // More threads than array size

    int *d_arr = NULL;
    int *h_arr = NULL;

    // Allocate host memory
    h_arr = (int*)malloc(N * sizeof(int));
    if (h_arr == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_arr, N * sizeof(int));

    // Launch kernel (intentionally causing out-of-bounds writes)
    out_of_bounds_kernel<<<1, THREADS_PER_BLOCK>>>(d_arr, N);

    // Copy result back to host (this may contain garbage due to out-of-bounds)
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results (not checking for errors)
    for (int i = 0; i < N; ++i) {
        printf("h_arr[%d] = %d\n", i, h_arr[i]);
    }

    // Free memory
    cudaFree(d_arr);
    free(h_arr);

    printf("Program completed.\n");
    return 0;
}
