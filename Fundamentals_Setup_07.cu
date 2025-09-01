```cpp
/*
Query and print the maximum dimensions (x, y, z) of a grid of thread blocks for the primary CUDA device. This shows the maximum number of blocks you can launch in each dimension.

Thinking:
- Use the CUDA runtime API to query the device properties of the current device (device 0).
- The relevant field is maxGridSize which is an array of three integers representing the maximum grid dimensions in x, y, and z.
- Use cudaGetDeviceProperties to fill a cudaDeviceProp structure.
- Check for errors in the API call and handle them appropriately.
- Print the maximum grid dimensions to standard output in a clear format.
- The program is a simple console application; it does not launch any kernels.
- Include necessary headers: <cstdio> for printf, <cuda_runtime.h> for CUDA API.
- Use a main function returning 0 on success.
- The code is minimal and self-contained, suitable for compiling with nvcc.
*/
#include <cstdio>
#include <cuda_runtime.h>

int main(void) {
    int device = 0; // Primary CUDA device
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Maximum grid dimensions for device %d (%s):\n", device, prop.name);
    printf("  x = %d\n", prop.maxGridSize[0]);
    printf("  y = %d\n", prop.maxGridSize[1]);
    printf("  z = %d\n", prop.maxGridSize[2]);

    return EXIT_SUCCESS;
}
```