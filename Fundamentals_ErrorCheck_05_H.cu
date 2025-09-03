```cpp
/*
Write a robust function `bool set_device(int id)` that checks the device count first and only calls `cudaSetDevice` if the ID is valid, returning `false` otherwise.

Thinking:
- The function must be robust, so it should first query the number of CUDA-capable devices on the system with cudaGetDeviceCount.
- If cudaGetDeviceCount fails, we cannot proceed; the function should return false.
- The id parameter should be checked against the device count: it must be non-negative and less than the count.
- If the id is invalid, return false without calling cudaSetDevice.
- When calling cudaSetDevice, we should check its return status. If it fails, return false.
- The function should return true only when all checks pass and cudaSetDevice succeeds.
- For demonstration purposes, a simple main() is provided that queries the device count, attempts to set device 0, and prints the result.
- All error handling uses cudaError_t and prints human-readable error messages to stderr.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

bool set_device(int id) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    if (id < 0 || id >= deviceCount) {
        fprintf(stderr, "Invalid device id %d. Available devices: 0 to %d.\n", id, deviceCount - 1);
        return false;
    }

    err = cudaSetDevice(id);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", id, cudaGetErrorString(err));
        return false;
    }

    return true;
}

int main(int argc, char *argv[]) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("CUDA device count: %d\n", deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    int testId = 0;
    if (argc > 1) {
        testId = atoi(argv[1]);
    }

    if (set_device(testId)) {
        printf("Successfully set device to %d.\n", testId);
    } else {
        printf("Failed to set device to %d.\n", testId);
        return EXIT_FAILURE;
    }

    // Optionally, we can query the current device to verify
    int currentDevice = -1;
    err = cudaGetDevice(&currentDevice);
    if (err == cudaSuccess) {
        printf("Current CUDA device is %d.\n", currentDevice);
    } else {
        fprintf(stderr, "Failed to get current device: %s\n", cudaGetErrorString(err));
    }

    return EXIT_SUCCESS;
}
```