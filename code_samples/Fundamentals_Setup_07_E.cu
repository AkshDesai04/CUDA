```cpp
/*
 * Aim: Query and print the device's UUID (Universally Unique ID).
 *
 * Thought process:
 * 1. Use CUDA Runtime API to enumerate devices with cudaGetDeviceCount.
 * 2. For each device, obtain its UUID via cudaDeviceGetUUID.
 * 3. uuid_t contains 16 bytes; format these into a hexadecimal string.
 * 4. Print the UUID for each device to standard output.
 * 5. Handle errors: if uuid is not supported on older devices, check return status.
 * 6. The program is written in C++ but uses CUDA Runtime headers and can be compiled with nvcc.
 * 7. Ensure the code is self‑contained, only includes necessary headers, and prints clear output.
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        uuid_t uuid;
        err = cudaDeviceGetUUID(&uuid, dev);
        if (err == cudaSuccess) {
            printf("Device %d UUID: ", dev);
            for (int i = 0; i < 16; ++i) {
                printf("%02x", uuid.data[i]);
                if (i == 3 || i == 5 || i == 7 || i == 9) {
                    printf("-");
                }
            }
            printf("\n");
        } else if (err == cudaErrorNoDevice) {
            printf("Device %d does not support UUID retrieval.\n", dev);
        } else {
            fprintf(stderr, "Failed to get UUID for device %d: %s\n", dev, cudaGetErrorString(err));
        }
    }

    return 0;
}
```