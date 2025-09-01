```c
/* 
 * Aim of the program: 
 * Write a program that attempts to set the device to an invalid index (e.g., 999) and use proper error checking to report that the device does not exist. 
 *
 * Thinking:
 * 1. The program is a simple CUDA host-only program; we don't launch any kernels. 
 * 2. We use the CUDA Runtime API, specifically cudaSetDevice() which takes an int device index and returns a cudaError_t.
 * 3. When an invalid device index is passed, cudaSetDevice() will return an error code such as cudaErrorInvalidDevice.
 * 4. We'll check the return value directly. If it's not cudaSuccess, we'll print a descriptive error message that includes the error string obtained from cudaGetErrorString(err).
 * 5. Additionally, to give the user context, we can query the number of available devices with cudaGetDeviceCount() and print that, so the user knows the valid range.
 * 6. The program will exit with a non-zero status if the device set failed, otherwise it will exit successfully (though in this case it will always fail because 999 is an invalid index).
 * 7. The code includes minimal error handling, uses printf for output, and is fully self-contained in a single .cu file.
 */
 
#include <stdio.h>
#include <cuda_runtime.h>
 
int main(void)
{
    int invalid_device = 999;
    cudaError_t err = cudaSetDevice(invalid_device);
 
    if (err != cudaSuccess) {
        int deviceCount = 0;
        cudaError_t countErr = cudaGetDeviceCount(&deviceCount);
        if (countErr == cudaSuccess) {
            printf("Failed to set device to index %d. Number of available devices: %d\n", invalid_device, deviceCount);
        } else {
            printf("Failed to set device to index %d. Additionally, could not retrieve device count.\n", invalid_device);
        }
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
 
    printf("Device %d set successfully (this should not happen for an invalid index).\n", invalid_device);
    return 0;
}
```