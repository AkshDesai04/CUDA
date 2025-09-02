/*
Aim of the program:
Does this error occur synchronously or asynchronously?

Thinking:
In CUDA, many operations are asynchronous with respect to the host. 
Kernel launches are asynchronous – the launch returns immediately and the 
kernel executes on the device concurrently. If a kernel causes an error 
(e.g., accessing memory out of bounds), the error is not reported to the 
host until a synchronous operation such as cudaGetLastError() or 
cudaDeviceSynchronize() is called. These functions perform a device-to-host 
synchronization, blocking the host until the device completes its work, and 
then they return the status of any errors that occurred during the device 
execution. Thus, the error itself is detected asynchronously, but the host 
side error reporting is synchronous. This program demonstrates that by 
launching a kernel that performs an out-of-bounds write, then checking the
error immediately (synchronously) after the launch and also after a 
synchronization point. The program prints whether an error was found in each
step, illustrating the difference between asynchronous kernel execution
and synchronous error retrieval.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that deliberately writes out of bounds
__global__ void out_of_bounds_kernel(int *d_arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Intentionally write beyond array bounds
    if (idx == 0) {
        // size is the actual allocated size, but we write to size (i.e., out of bounds)
        d_arr[size] = 42;  // This will cause a memory error
    }
}

int checkCudaError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after %s: %s\n", msg, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int main(void) {
    const int N = 10;
    int *d_arr = NULL;

    // Allocate device memory
    cudaMalloc((void**)&d_arr, N * sizeof(int));
    // Initialize to zero
    cudaMemset(d_arr, 0, N * sizeof(int));

    // Launch kernel that writes out of bounds
    out_of_bounds_kernel<<<1, 1>>>(d_arr, N);

    // Immediately check for errors (synchronously)
    if (checkCudaError("kernel launch")) {
        // If error was detected, exit early
        cudaFree(d_arr);
        return 1;
    } else {
        printf("No error detected immediately after kernel launch.\n");
    }

    // Now perform an explicit device synchronization
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "Error during cudaDeviceSynchronize: %s\n", cudaGetErrorString(sync_err));
        cudaFree(d_arr);
        return 1;
    } else {
        printf("cudaDeviceSynchronize completed without error.\n");
    }

    // After synchronization, check for errors again
    if (checkCudaError("after synchronize")) {
        // Error was caught after synchronization
        cudaFree(d_arr);
        return 1;
    } else {
        printf("No error detected after cudaDeviceSynchronize.\n");
    }

    // Clean up
    cudaFree(d_arr);
    printf("Program finished successfully.\n");
    return 0;
}
