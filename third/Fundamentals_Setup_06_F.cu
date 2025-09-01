/*
Print the maximum z-dimension of a grid of blocks.
The program queries the CUDA device properties for the currently active
device (defaulting to device 0). It retrieves the maximum grid size
information stored in cudaDeviceProp.maxGridSize, which is an array of
three integers representing the maximum allowed dimensions for the
grid in the x, y, and z directions respectively. The program prints
the value corresponding to the z-dimension (index 2). Error checking
is performed for all CUDA runtime API calls; if an error occurs, an
error message is printed to stderr and the program exits with a
non‑zero status. This simple demonstration showcases how to access
hardware limits from CUDA C code.

*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device = 0;                     // Default to device 0
    cudaDeviceProp prop;
    cudaError_t err;

    // Get the current device (in case the user changed it)
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Retrieve properties of the selected device
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print the maximum z-dimension for the grid
    printf("Maximum grid z-dimension: %d\n", prop.maxGridSize[2]);

    return 0;
}
