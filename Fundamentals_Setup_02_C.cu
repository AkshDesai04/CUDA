```c
/*
Aim of the program: Iterate through all devices and print the warp size for each one.

Thinking:
To accomplish the goal we need to query the system for the number of CUDA-capable devices,
then for each device retrieve its properties using cudaGetDeviceProperties. The warp size
is stored in the 'warpSize' member of the cudaDeviceProp structure. We will print each
device index along with its warp size. Basic error handling will be added to ensure that
any CUDA API call failures are reported. The program will be written in plain C with
CUDA runtime API calls and will compile with nvcc. No additional functionality beyond
printing the warp size is required, but printing the device name could be helpful for
clarity, though it's optional.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
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

    printf("Found %d CUDA device(s).\n", deviceCount);
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }
        printf("Device %d (%s): Warp Size = %d\n", dev, prop.name, prop.warpSize);
    }

    return 0;
}
```