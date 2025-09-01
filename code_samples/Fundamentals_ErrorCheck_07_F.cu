```cuda
/*
Aim: Write code that triggers an asynchronous error. Show that the API call itself returns `cudaSuccess` but `cudaDeviceSynchronize` later returns an error.

Thinking:
To demonstrate that CUDA kernel launches are asynchronous, I will create a kernel that writes to an array on the device. 
The kernel will be launched with one more thread than the size of the allocated array, so the last thread writes outside the bounds of the array. 
This out‑of‑bounds access is an illegal memory operation that causes an error, but because the launch is asynchronous the error is not reported immediately. 
After the launch I will call cudaGetLastError(), which should return cudaSuccess because the error has not yet been propagated. 
Then I will call cudaDeviceSynchronize() to force the device to finish all pending work. 
At that point the runtime will detect the illegal memory access and return an error code (e.g. cudaErrorInvalidDevicePointer or cudaErrorLaunchFailure). 
I will print the error string before and after synchronization to illustrate the difference. 
Finally, I will clean up the allocated memory and reset the device. 
This example shows clearly that a kernel launch can return cudaSuccess even though it will later fail during synchronization.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void write_out_of_bounds(int *arr, int size)
{
    // Every thread writes to its own index. 
    // The kernel will be launched with size+1 threads, so the last thread writes outside the array.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    arr[idx] = idx;  // out‑of‑bounds write for idx == size
}

int main()
{
    const int N = 10;          // Size of the array on the device
    int *d_arr = nullptr;

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_arr, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Launch the kernel with one more thread than the array size
    const int threadsPerBlock = 256;
    const int blocks = (N + 1 + threadsPerBlock - 1) / threadsPerBlock;
    write_out_of_bounds<<<blocks, threadsPerBlock>>>(d_arr, N);

    // Check error immediately after launch (should be cudaSuccess)
    err = cudaGetLastError();
    printf("After kernel launch, cudaGetLastError: %s\n", cudaGetErrorString(err));

    // Now synchronize to force the device to finish execution
    err = cudaDeviceSynchronize();
    printf("After cudaDeviceSynchronize, error: %s\n", cudaGetErrorString(err));

    // Optionally reset the device to clear any errors
    cudaDeviceReset();

    // Free device memory
    cudaFree(d_arr);

    return 0;
}
```