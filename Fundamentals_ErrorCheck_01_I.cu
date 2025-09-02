/*
Aim of the program:
Write the error checking logic in a helper function `void check_error(cudaError_t code, const char* file, int line)` and have the macro call this function.

Thinking:
- The core requirement is to provide a function that checks the CUDA API return codes and prints an informative error message with the file name and line number where the error occurred. If an error is detected, the program should terminate gracefully.
- A convenient macro is needed so that the programmer can call `CUDA_CHECK` with a CUDA API function call, and the macro will expand to a call to `check_error` passing `__FILE__` and `__LINE__`.
- To demonstrate the usage of the error checking mechanism, a minimal CUDA kernel and a simple host program are added. The host code performs memory allocation, launches the kernel, checks for errors after the launch, synchronizes the device, and frees the memory, all wrapped with `CUDA_CHECK`.
- The program uses the CUDA runtime API, so `<cuda_runtime.h>` and `<stdio.h>` are included.
- The `check_error` function prints the error string obtained from `cudaGetErrorString` and exits with `EXIT_FAILURE` if the code is not `cudaSuccess`.
- The macro `CUDA_CHECK` takes a single argument, which is the CUDA API call or `cudaGetLastError()`, and passes the file and line to `check_error`.

This file is self‑contained and can be compiled with `nvcc` to test the error checking logic.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* Helper function to check CUDA errors */
void check_error(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error [%s] at %s:%d\n",
                cudaGetErrorString(code), file, line);
        exit(EXIT_FAILURE);
    }
}

/* Macro that calls the error checking helper */
#define CUDA_CHECK(code) check_error((code), __FILE__, __LINE__)

/* Simple kernel for demonstration */
__global__ void dummy_kernel(int *d_arr) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d_arr[idx] = idx;
}

int main(void) {
    const int N = 256;
    int *d_arr = NULL;

    /* Allocate device memory */
    CUDA_CHECK(cudaMalloc(&d_arr, N * sizeof(int)));

    /* Launch kernel */
    dummy_kernel<<<(N + 255) / 256, 256>>>(d_arr);

    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Wait for kernel to finish */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Free device memory */
    CUDA_CHECK(cudaFree(d_arr));

    printf("Kernel executed successfully.\n");
    return 0;
}
