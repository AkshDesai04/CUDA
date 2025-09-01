/*
Aim: Try to launch a kernel using a function pointer that is null. What error occurs?

Thinking: 
- The program demonstrates that if we attempt to launch a kernel through a null function pointer, the CUDA runtime will report an error during the launch.
- A simple dummy kernel is defined. We create a function pointer to that kernel type but set it to nullptr.
- We then use the function pointer to launch the kernel: kernelPtr<<<1,1>>>().
- Since kernelPtr is null, the launch will fail. After the launch, we call cudaGetLastError() to obtain the error code and print the corresponding error string.
- The error typically reported is `cudaErrorLaunchFailure` (code 7) or `cudaErrorInvalidDeviceFunction` (code 13) depending on the runtime version. In this example we will capture and display whatever error the runtime reports.
- The program includes basic error handling and prints the error after the launch attempt.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Dummy kernel (does nothing)
__global__ void dummyKernel()
{
    // Intentionally left empty
}

int main()
{
    // Function pointer to a __global__ function, initialized to nullptr
    __global__ void (*kernelPtr)() = nullptr;

    // Attempt to launch the kernel through the null pointer
    kernelPtr<<<1, 1>>>();

    // Synchronize to force the kernel launch to complete (or fail)
    cudaError_t syncErr = cudaDeviceSynchronize();

    // Get the error from the launch
    cudaError_t launchErr = cudaGetLastError();

    // Print results
    printf("cudaDeviceSynchronize error: %s\n", cudaGetErrorString(syncErr));
    printf("cudaGetLastError after launch: %s\n", cudaGetErrorString(launchErr));

    // Clean up (none needed for this simple example)
    return 0;
}
