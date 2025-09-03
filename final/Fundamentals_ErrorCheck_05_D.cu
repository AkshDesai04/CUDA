/*
Write a program that checks `cudaDeviceCanAccessPeer` with an invalid peer device index.

I started by planning a simple CUDA C program that will:
1. Query the number of CUDA devices with cudaGetDeviceCount.
2. If no devices are present, print an error and exit.
3. Select device 0 for the test using cudaSetDevice.
4. Construct an invalid peer index. The easiest way to guarantee an invalid index is to use a value equal to the device count (since valid indices range from 0 to count-1). I also considered negative indices but using count works on all systems.
5. Call cudaDeviceCanAccessPeer with the selected device (0) and the invalid peer index.
6. The function returns a cudaError_t; I will check if it equals cudaSuccess. For an invalid peer it should return a non-success error code such as cudaErrorInvalidDevice. I will print the returned error code and the corresponding error string using cudaGetErrorString for clarity.
7. Finally, exit gracefully.

The program includes basic error checking for CUDA calls and prints informative messages so that the user sees what went wrong when the peer index is invalid.

*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err;

    // Get number of CUDA-capable devices
    err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return 1;
    }

    // Select device 0
    int dev = 0;
    err = cudaSetDevice(dev);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", dev, cudaGetErrorString(err));
        return 1;
    }

    // Construct an invalid peer index (deviceCount is out of range)
    int invalidPeer = deviceCount; // out-of-range index

    // Check peer access
    bool canAccess;
    err = cudaDeviceCanAccessPeer(&canAccess, dev, invalidPeer);
    if (err != cudaSuccess) {
        printf("cudaDeviceCanAccessPeer returned error for invalid peer index %d: %s\n",
               invalidPeer, cudaGetErrorString(err));
    } else {
        printf("Unexpected success: cudaDeviceCanAccessPeer returned true for invalid peer index %d.\n",
               invalidPeer);
    }

    return 0;
}
