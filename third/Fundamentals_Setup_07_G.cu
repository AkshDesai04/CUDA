```c
/*
 * Aim of the program: Print the `major` and `minor` compute capability numbers for the primary device.
 *
 * Thinking:
 * 1. To obtain the compute capability of the current primary CUDA device, we need to query the device properties.
 * 2. The CUDA Runtime API provides `cudaGetDevice` to get the current device id, and `cudaGetDeviceProperties`
 *    to retrieve a `cudaDeviceProp` structure containing information about the device, including the major
 *    and minor compute capability fields.
 * 3. We'll call `cudaGetDevice(&dev)` to get the device id, then `cudaGetDeviceProperties(&prop, dev)` to fill the
 *    properties struct. After that, we simply print `prop.major` and `prop.minor`.
 * 4. Error handling is minimal; we just check the return status of the CUDA API calls and exit with a message if
 *    something goes wrong.
 * 5. The output will be a simple text line: "Compute capability: X.Y" where X and Y are the major and minor numbers.
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device = 0;
    cudaError_t err;

    // Get the current device
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get current device: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Retrieve device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Print the compute capability
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);

    return 0;
}
```