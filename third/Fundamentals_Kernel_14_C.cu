/*
Use `cudaPeekAtLastError()` immediately after the kernel launch to get the error code without waiting for a synchronization.

Thinking:
The goal is to illustrate how `cudaPeekAtLastError()` can be used right after a kernel launch to capture any error that occurred during the launch itself, without having to perform a synchronization or an explicit `cudaDeviceSynchronize()`.  
A kernel launch can fail for various reasons, the most common being an illegal configuration (e.g., requesting more threads per block than the device supports).  
To trigger such a failure, we query the device properties to find the maximum number of threads per block (`maxThreadsPerBlock`) and then intentionally launch the kernel with twice that number of threads.  
The kernel is trivial – it does nothing – because the failure occurs before the kernel executes.  
After the launch we immediately call `cudaPeekAtLastError()`. This function returns the error code from the last runtime API call (the kernel launch), and does not wait for the device to finish executing the kernel.  
If the launch succeeded, we print a success message; otherwise we print the error string.  
For completeness we also call `cudaDeviceSynchronize()` afterwards to check for any runtime errors that might occur during kernel execution (though in this case the launch itself has already failed, so the synchronize will also report the error).
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dummyKernel()
{
    // Simple kernel that does nothing
}

int main()
{
    // Get device properties
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        printf("Failed to get device properties: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

    // Intentionally exceed the maximum threads per block to trigger a launch error
    int threadsPerBlock = prop.maxThreadsPerBlock * 2; // Exceeds device limit
    dummyKernel<<<1, threadsPerBlock>>>();

    // Immediately peek at the last error after kernel launch
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error (peek): %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel launch succeeded (peek).\n");
    }

    // Optional: synchronize to see if any runtime errors occur during execution
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution error (sync): %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel executed successfully (sync).\n");
    }

    return 0;
}
