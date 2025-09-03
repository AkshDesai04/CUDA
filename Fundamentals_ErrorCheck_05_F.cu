```c
/*
 * Aim of the program: What happens if you have 0 CUDA devices and you try to cudaSetDevice(0)?
 *
 * Thinking process:
 * 1. The program aims to demonstrate the behavior of the CUDA runtime when there are no devices
 *    available on the system and the user attempts to set a device by calling cudaSetDevice(0).
 * 2. We need to query the number of CUDA-capable devices using cudaGetDeviceCount. If the count
 *    is zero, we still attempt to set device 0.
 * 3. The CUDA runtime will return an error code (cudaErrorNoDevice) indicating that there are
 *    no devices to set. We capture this error and print it using cudaGetErrorString to provide
 *    a human-readable description.
 * 4. The code is written in plain C using CUDA runtime API functions. It is self‑contained and
 *    can be compiled with nvcc. No external libraries or additional files are required.
 * 5. The program also handles the case when devices are present; it sets device 0 and reports
 *    success, to illustrate the normal path.
 * 6. Error handling is performed after each CUDA call by checking the returned cudaError_t.
 * 7. The output of the program will clearly indicate whether cudaSetDevice succeeded or failed
 *    and why, demonstrating the effect of having zero CUDA devices.
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error querying device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("CUDA device count: %d\n", deviceCount);

    // Attempt to set device 0 regardless of device count
    printf("Attempting to set CUDA device 0...\n");
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printf("cudaSetDevice(0) failed: %s (error code %d)\n", cudaGetErrorString(err), err);
    } else {
        printf("cudaSetDevice(0) succeeded.\n");
    }

    // If a device was successfully set, we can perform a simple query to confirm
    if (err == cudaSuccess) {
        int dev;
        err = cudaGetDevice(&dev);
        if (err == cudaSuccess) {
            printf("Currently active device: %d\n", dev);
        } else {
            printf("Failed to get current device: %s\n", cudaGetErrorString(err));
        }
    }

    return 0;
}
```