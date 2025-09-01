```cuda
/*
 * Aim: Does `cudaDeviceSynchronize` need to be wrapped? Yes, it can return asynchronous errors.
 *
 * Thinking:
 * 1. The purpose of this sample is to illustrate how asynchronous errors in CUDA kernels
 *    can be detected only when we invoke `cudaDeviceSynchronize()` (or a wrapper that
 *    checks its return value). If we rely solely on `cudaGetLastError()` immediately
 *    after kernel launch, we might miss errors that occur during kernel execution
 *    because those errors are reported only when the device reaches the point of
 *    synchronization.
 *
 * 2. The code below performs a deliberately incorrect memory write inside a kernel
 *    (writing outside the bounds of a device array). This triggers an asynchronous
 *    error (e.g., `cudaErrorInvalidDevicePointer` or `cudaErrorIllegalAddress`). We
 *    launch the kernel and then:
 *      - call `cudaGetLastError()` right after the launch to see if any synchronous
 *        errors are reported (none expected).
 *      - call `cudaDeviceSynchronize()` and check its return status. This should
 *        catch the asynchronous error and allow us to print an appropriate message.
 *
 * 3. The wrapper function `checkCudaError()` is defined to wrap any CUDA call and
 *    print an error message if the call fails. It is used for both the device
 *    memory allocation and the synchronization step.
 *
 * 4. This sample is intentionally simple, using only standard CUDA APIs and plain
 *    C code. It can be compiled with `nvcc` and demonstrates the necessity of
 *    checking the return value of `cudaDeviceSynchronize()` to catch asynchronous
 *    kernel errors.
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* Macro to check CUDA errors */
#define CHECK_CUDA(call)                                       \
    do {                                                       \
        cudaError_t err = call;                                \
        if (err != cudaSuccess) {                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

/* Simple kernel that intentionally writes out-of-bounds to trigger an error */
__global__ void badKernel(int *d_arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    /* Intentionally access out-of-bounds when idx == size */
    if (idx <= size) {
        d_arr[idx] = idx;
    }
}

int main(void) {
    const int N = 5;
    int *d_arr = NULL;

    /* Allocate device memory */
    CHECK_CUDA(cudaMalloc((void**)&d_arr, N * sizeof(int)));

    /* Launch kernel with a block of 256 threads */
    badKernel<<<1, 256>>>(d_arr, N);

    /* Check for synchronous errors after launch */
    cudaError_t sync_err = cudaGetLastError();
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "Synchronous error after kernel launch: %s\n",
                cudaGetErrorString(sync_err));
    } else {
        printf("No synchronous errors reported after kernel launch.\n");
    }

    /* Wrap cudaDeviceSynchronize to catch asynchronous errors */
    sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "Asynchronous error detected by cudaDeviceSynchronize: %s\n",
                cudaGetErrorString(sync_err));
    } else {
        printf("Kernel executed successfully. No asynchronous errors.\n");
    }

    /* Clean up */
    CHECK_CUDA(cudaFree(d_arr));
    return 0;
}
```