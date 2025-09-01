/*
Write code that triggers a synchronous error and use the `CHECK` macro to catch it immediately.

Thinking:
In CUDA, kernel launches are asynchronous; the error from a bad configuration is reported on the next API call.  
To catch it synchronously, we call `CHECK(cudaGetLastError())` immediately after the launch.  
The macro will print the error and exit.  
The kernel is launched with too many threads per block, exceeding the device limit, which triggers an
`invalid configuration argument` error.  This is a typical synchronous error catch scenario.  

We also demonstrate usage of `CHECK` for other API calls like `cudaMalloc`, `cudaFree`, and
`cudaDeviceSynchronize`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Macro to check CUDA errors */
#define CHECK(call)                                                       \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d - %s (%d)\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err), err);   \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

/* Simple kernel that writes to an array */
__global__ void dummy_kernel(int *d_arr) {
    int idx = threadIdx.x;
    d_arr[idx] = idx;
}

int main(void) {
    int *d_arr;
    size_t size = 5 * sizeof(int);

    /* Allocate device memory */
    CHECK(cudaMalloc(&d_arr, size));

    /* Launch kernel with an invalid configuration:
       Exceed the maximum number of threads per block (usually 1024) */
    dummy_kernel<<<1, 2048>>>(d_arr);

    /* Immediately check for kernel launch errors synchronously */
    CHECK(cudaGetLastError());

    /* If no error, synchronize and finish */
    CHECK(cudaDeviceSynchronize());

    /* Clean up */
    CHECK(cudaFree(d_arr));

    printf("Program completed without detecting a synchronous error.\n");
    return 0;
}
