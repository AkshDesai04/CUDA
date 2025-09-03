```cuda
/*
In the version with error checking, use your `CHECK` macro to catch and report the `cudaErrorIllegalAddress`.

I am going to create a macro named CHECK that checks the result of any CUDA runtime API call. I will then write a kernel that deliberately writes outside the bounds of an allocated array to trigger cudaErrorIllegalAddress. After launching the kernel I will call CHECK(cudaGetLastError()) to catch the error. The macro will print the error string and exit. The code will allocate memory, launch kernel, check error, free memory, and return 0. This demonstrates using the macro to catch the illegal address error. */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA runtime API calls */
#define CHECK(call)                                                \
    do {                                                           \
        cudaError_t err = (call);                                  \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr,                                         \
                    "CUDA error at %s:%d code=%d \"%s\"\\n",       \
                    __FILE__, __LINE__, (int)err,                 \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

/* Kernel that intentionally accesses an illegal address */
__global__ void illegal_kernel(int *d_data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        /* This write is out of bounds (size is the valid last index) */
        d_data[size] = 42; /* triggers cudaErrorIllegalAddress */
    }
}

int main(void)
{
    const int N = 256;
    int *d_data = NULL;

    /* Allocate device memory */
    CHECK(cudaMalloc((void **)&d_data, N * sizeof(int)));
    CHECK(cudaMemset(d_data, 0, N * sizeof(int)));

    /* Launch kernel with one thread */
    illegal_kernel<<<1, 1>>>(d_data, N);

    /* Catch any errors from the kernel launch */
    CHECK(cudaGetLastError());

    /* Clean up */
    CHECK(cudaFree(d_data));

    printf("Program completed without detecting an illegal address (unexpected).\n");
    return 0;
}
```