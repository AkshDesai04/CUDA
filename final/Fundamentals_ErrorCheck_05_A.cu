/*
Aim of the program: First, get the device count. Then, try to set the device to `count` (which is an invalid index, since they are 0-based).

My thinking:
- The goal is to demonstrate the use of cudaGetDeviceCount to obtain the number of CUDA-capable devices on the system.
- After retrieving the count, we will deliberately pass this count value to cudaSetDevice. Since device indices are 0-based, a value equal to the count is out of range and should produce a cudaErrorInvalidDevice.
- We'll handle the return values of both CUDA API calls, printing clear error messages if anything goes wrong.
- The program will be written in standard C with CUDA runtime API, and compiled with nvcc as a .cu file.
- No kernel launch or device memory allocation is necessary; the focus is purely on device enumeration and error handling.
- We include <cuda_runtime.h> and <stdio.h> for the API and printing.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving device count: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    // Intentionally use an invalid device index (equal to deviceCount)
    printf("Attempting to set device to index %d (invalid if count > 0)...\n", deviceCount);
    err = cudaSetDevice(deviceCount);
    if (err != cudaSuccess) {
        printf("cudaSetDevice failed as expected: %s\n", cudaGetErrorString(err));
        // Optionally exit with failure to indicate expected error occurred
        return EXIT_FAILURE;
    } else {
        // If no error, this is unexpected
        printf("cudaSetDevice succeeded unexpectedly. This indicates a bug or no devices present.\n");
    }

    // If we reach here, we might want to reset to a valid device before exiting
    if (deviceCount > 0) {
        err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error setting to device 0 during cleanup: %s\n", cudaGetErrorString(err));
            return EXIT_FAILURE;
        }
    }

    printf("Program completed.\n");
    return EXIT_SUCCESS;
}
