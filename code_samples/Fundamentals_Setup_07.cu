```cpp
/*
 * Aim of the program: Query and print the maximum dimensions (x, y, z) of a grid of thread blocks for the primary CUDA device. This shows the maximum number of blocks you can launch in each dimension.
 *
 * Thinking:
 * 1. Use cudaGetDeviceProperties to retrieve the properties of the current device. This structure contains fields maxGridSize[3] that represent the maximum grid dimensions for x, y, and z.
 * 2. Alternatively, cudaDeviceGetAttribute can retrieve each dimension individually, but using the device properties struct is simpler and provides all three in one call.
 * 3. Set the current device to the primary device (device 0) using cudaSetDevice(0) to be explicit.
 * 4. Perform error checking after each CUDA API call. If an error occurs, print an error message and exit.
 * 5. Print the results in a human‑readable format.
 * 6. The program is self‑contained, requires only the CUDA runtime header, and compiles with nvcc as a .cu file.
 * 7. The code uses standard C++ I/O via printf from <stdio.h> for simplicity.
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    cudaError_t err;

    // Set the device to the primary CUDA device (device 0)
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Retrieve properties of the current device
    cudaDeviceProp devProp;
    err = cudaGetDeviceProperties(&devProp, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print maximum grid dimensions
    printf("Maximum grid dimensions for device %d (%s):\n", 0, devProp.name);
    printf("  x: %d\n", devProp.maxGridSize[0]);
    printf("  y: %d\n", devProp.maxGridSize[1]);
    printf("  z: %d\n", devProp.maxGridSize[2]);

    return 0;
}
```