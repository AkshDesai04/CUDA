/*
Aim of the program: Explain the difference between synchronous and asynchronous CUDA errors. Write a code snippet that would cause a synchronous error (e.g., invalid cudaMalloc size) and another that would cause an asynchronous error (e.g., out-of-bounds access inside a kernel).
Thinking:
- A synchronous error is reported immediately by the CUDA runtime function that triggered it. For example, cudaMalloc will return cudaErrorInvalidValue if the requested size is too large or negative. The error code is available right after the call.
- An asynchronous error occurs during the execution of a kernel or an operation that is queued on the GPU. The error is not reported until a subsequent CUDA API call or cudaDeviceSynchronize is executed. Typical examples are out-of-bounds memory accesses or division by zero inside a kernel.
- The program below demonstrates both types of errors. The first part attempts to allocate an absurdly large amount of memory, which fails synchronously. The second part launches a kernel that writes past the bounds of its device array; this failure is asynchronous and will only be detected when we call cudaGetLastError() or cudaDeviceSynchronize().
- We print the error messages to illustrate the difference. In a real application, error checking after every CUDA call is essential.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Kernel that intentionally writes out of bounds */
__global__ void out_of_bounds_kernel(int *data, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    /* Intentionally access beyond the allocated array */
    if (idx < size)
    {
        data[idx] = idx;      // valid part
    }
    /* Out-of-bounds write */
    data[size + idx] = -1;   // This will cause an async error
}

int main()
{
    /* ----------------- Synchronous Error Example ----------------- */
    size_t huge_size = (size_t)2 * 1024 * 1024 * 1024 * 1024; // 2 TB
    int *dev_ptr = NULL;
    cudaError_t err = cudaMalloc((void**)&dev_ptr, huge_size);
    if (err != cudaSuccess)
    {
        printf("Synchronous error (cudaMalloc): %s\n", cudaGetErrorString(err));
    }
    else
    {
        // Should not reach here
        cudaFree(dev_ptr);
    }

    /* ----------------- Asynchronous Error Example ----------------- */
    const int N = 10;
    int *d_array = NULL;
    err = cudaMalloc((void**)&d_array, N * sizeof(int));
    if (err != cudaSuccess)
    {
        printf("Failed to allocate d_array: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Launch kernel with a small number of threads, but the kernel writes out-of-bounds
    out_of_bounds_kernel<<<1, 5>>>(d_array, N);

    // At this point, no error is reported because the kernel launch is asynchronous
    err = cudaGetLastError(); // Checks for launch errors
    if (err != cudaSuccess)
    {
        printf("Asynchronous error (kernel launch): %s\n", cudaGetErrorString(err));
    }

    // Synchronize to force the kernel to finish and propagate any runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("Asynchronous error (kernel execution): %s\n", cudaGetErrorString(err));
    }
    else
    {
        printf("Kernel executed without errors (this should not happen).\n");
    }

    cudaFree(d_array);
    return 0;
}
