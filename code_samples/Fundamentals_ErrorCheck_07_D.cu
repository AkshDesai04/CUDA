```cuda
/*
How are asynchronous errors reported? (Via a subsequent synchronous call or `cuda...Synchronize`).

Thinking:
In CUDA, errors that occur during kernel launch are reported synchronously
via cudaGetLastError() immediately after the launch. However, errors that
occur during the actual execution of the kernel (for example, accessing
out-of-bounds memory, division by zero, or illegal instructions) are
reported asynchronously. To capture those, one must perform a
synchronous operation such as cudaDeviceSynchronize() or any other
cudaStreamSynchronize() call. The error will then be returned by that
synchronization function or by a subsequent cudaGetLastError() call.
This program demonstrates both mechanisms by launching a kernel that
writes out of bounds (causing an asynchronous error). We check for
errors using cudaGetLastError() right after the launch and then again
after cudaDeviceSynchronize().
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that writes past the bounds of the array
__global__ void out_of_bounds_kernel(int *d_arr, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        // Intentionally write out of bounds when idx is size-1
        d_arr[idx] = idx;          // valid
        d_arr[idx + 1] = idx + 1;  // out of bounds write
    }
}

int checkCudaError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int main(void)
{
    const int N = 16;
    const size_t bytes = N * sizeof(int);

    int *d_arr = NULL;
    cudaMalloc((void **)&d_arr, bytes);

    // Launch kernel that will cause an asynchronous out-of-bounds error
    out_of_bounds_kernel<<<1, N>>>(d_arr, N);

    // Check for synchronous errors (e.g., launch failures)
    if (checkCudaError("Error after kernel launch (synchronous)") != 0)
    {
        cudaFree(d_arr);
        return -1;
    }

    // At this point, the kernel may still be running asynchronously.
    // The out-of-bounds write will be detected when we synchronize.
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize error (asynchronous): %s\n",
                cudaGetErrorString(syncErr));
    }
    else
    {
        printf("cudaDeviceSynchronize succeeded, no asynchronous errors detected.\n");
    }

    // Clean up
    cudaFree(d_arr);
    return 0;
}
```