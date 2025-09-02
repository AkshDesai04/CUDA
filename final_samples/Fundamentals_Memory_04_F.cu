/*
Write a function that takes a device pointer and a size and calls `cudaMemset` to zero it out.
Thinking:
- The program needs to include the CUDA runtime header to access cudaMemset and related functions.
- We'll define a function with the signature `void zeroDeviceMemory(void* devPtr, size_t size)`.
- Inside the function, we'll call `cudaMemset(devPtr, 0, size)` and capture the returned cudaError_t.
- If an error occurs, we'll print a message using `fprintf` to `stderr`.
- For demonstration purposes, a `main` function is included that allocates device memory, zeros it using the function, copies the data back to the host, and verifies that all bytes are zero.
- The program is written in CUDA C (.cu file) and can be compiled with `nvcc`.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void zeroDeviceMemory(void* devPtr, size_t size) {
    cudaError_t err = cudaMemset(devPtr, 0, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(err));
    }
}

/* Optional test harness */
int main() {
    const size_t N = 1024;
    int* d_array = NULL;
    int* h_array = (int*)malloc(N * sizeof(int));

    /* Allocate device memory */
    cudaError_t err = cudaMalloc((void**)&d_array, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_array);
        return EXIT_FAILURE;
    }

    /* Zero the device memory */
    zeroDeviceMemory(d_array, N * sizeof(int));

    /* Copy back to host */
    err = cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_array);
        free(h_array);
        return EXIT_FAILURE;
    }

    /* Verify all elements are zero */
    int allZero = 1;
    for (size_t i = 0; i < N; ++i) {
        if (h_array[i] != 0) {
            allZero = 0;
            printf("Element %zu is not zero: %d\n", i, h_array[i]);
            break;
        }
    }

    if (allZero) {
        printf("All %zu elements are zero as expected.\n", N);
    } else {
        printf("Verification failed.\n");
    }

    /* Clean up */
    cudaFree(d_array);
    free(h_array);
    return EXIT_SUCCESS;
}
