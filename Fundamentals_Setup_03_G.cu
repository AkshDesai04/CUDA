/*
Aim: Query and print the maximum dimension of a 1D texture for device 0.

Thinking:
1. The program must use the CUDA Runtime API to query device properties.
2. First, obtain the number of CUDA devices using `cudaGetDeviceCount`.
3. Ensure at least one device exists; otherwise, report an error.
4. Use device ID 0 as requested.
5. Retrieve the device properties via `cudaGetDeviceProperties`.
6. The maximum 1D texture dimension is stored in the `maxTexture1D` member of `cudaDeviceProp`.
7. Print the result in a human-readable form, including the device name for clarity.
8. Handle errors by printing descriptive messages and exiting with a non-zero status.
9. Include necessary headers (`cuda_runtime.h`, `cstdio`) and use `int main` as the entry point.
10. Compile this code with `nvcc` to generate a runnable executable.

The final code below follows these steps and is self-contained.
*/

#include <cstdio>
#include <cuda_runtime.h>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("Error: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return -1;
    }

    int device = 0; // target device 0
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        printf("Error: cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Maximum 1D texture size on device %d (%s): %d\n",
           device, prop.name, prop.maxTexture1D);

    return 0;
}
