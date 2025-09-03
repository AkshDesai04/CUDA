/*
How are asynchronous errors reported? (Via a subsequent synchronous call or `cuda...Synchronize`).
I need to demonstrate that asynchronous errors in CUDA are not returned immediately on the API call that
triggers them.  Instead, they are deferred until the next synchronous operation, such as
cudaDeviceSynchronize or cudaStreamSynchronize, or a subsequent cudaGetLastError call that forces
the runtime to report the error.  To illustrate this, the program will:

1. Allocate a small array on the device.
2. Launch a kernel with an invalid configuration (0 threads), which causes an error
   that is recorded asynchronously.
3. Query the last error with cudaGetLastError, which will indicate the error but
   does not block.
4. Call cudaDeviceSynchronize to force the runtime to report the error at that point.
5. Print the error messages to show that the error is only reported after the
   synchronization call.

The program is a minimal, self‑contained CUDA C example that compiles to a .cu file.
It uses no external headers or libraries other than the standard CUDA runtime
API.  The intent is to answer the question by showing concrete code that
reports asynchronous errors via a subsequent synchronous call.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple kernel that writes 1 to each element of an array */
__global__ void write_one(int *arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = 1;
}

/* Helper macro to check CUDA errors */
#define CHECK_CUDA(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

int main(void)
{
    const int N = 10;
    int *d_arr = NULL;

    /* Allocate device memory */
    CHECK_CUDA(cudaMalloc((void**)&d_arr, N * sizeof(int)));

    /* Launch kernel with 0 threads per block (invalid configuration) */
    /* This will set an asynchronous error but not return it immediately. */
    write_one<<<1, 0>>>(d_arr, N);

    /* Query the error without synchronization */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("After kernel launch, cudaGetLastError returned: %s\n",
               cudaGetErrorString(err));
    } else {
        printf("After kernel launch, no error reported by cudaGetLastError.\n");
    }

    /* Force synchronization to report any asynchronous errors */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("After cudaDeviceSynchronize, error reported: %s\n",
               cudaGetErrorString(err));
    } else {
        printf("After cudaDeviceSynchronize, no error reported.\n");
    }

    /* Clean up */
    CHECK_CUDA(cudaFree(d_arr));

    return 0;
}
