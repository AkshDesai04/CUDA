/*
Look up the documentation for `cudaErrorInvalidDevice`. What other functions can return this error?

My thinking: The user asked for a textual answer, but the system instruction requires me to output a CUDA C program (.cu file). Therefore, I will create a small demo program that intentionally triggers `cudaErrorInvalidDevice` through various CUDA Runtime API calls. The comment above the code will contain the exact phrase from the prompt and also a description of my thought process. The program will:
1. Attempt to set an invalid device index with `cudaSetDevice`.
2. Attempt to enable peer access between an invalid device and a valid device with `cudaDeviceEnablePeerAccess`.
3. Check the returned error codes and print them, demonstrating where `cudaErrorInvalidDevice` can arise.
This satisfies the system instruction to provide a complete .cu file while also addressing the user's question by illustrating real functions that return this error.

*/

#include <stdio.h>
#include <cuda_runtime.h>

void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess) {
        printf("%s: %s\n", msg, cudaGetErrorString(err));
    } else {
        printf("%s: Success\n", msg);
    }
}

int main()
{
    cudaError_t err;

    // 1. Valid device query
    int deviceCount;
    err = cudaGetDeviceCount(&deviceCount);
    checkCudaError(err, "cudaGetDeviceCount");

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    // 2. Attempt to set an invalid device index
    int invalidDevice = deviceCount + 10; // deliberately out of range
    err = cudaSetDevice(invalidDevice);
    checkCudaError(err, "cudaSetDevice (invalid device)");

    // 3. Attempt to enable peer access with an invalid device
    int srcDevice = 0;
    int dstDevice = invalidDevice; // out of range
    err = cudaSetDevice(srcDevice);
    checkCudaError(err, "cudaSetDevice (source device)");

    err = cudaDeviceEnablePeerAccess(dstDevice, 0);
    checkCudaError(err, "cudaDeviceEnablePeerAccess (invalid dst device)");

    // 4. Attempt to set a negative device index
    int negativeDevice = -1;
    err = cudaSetDevice(negativeDevice);
    checkCudaError(err, "cudaSetDevice (negative device)");

    return 0;
}
