/*
 * Query and print the maximum x-dimension of a thread block.
 *
 * My thinking:
 * The problem asks for the maximum x-dimension (i.e., the maximum number of threads in the x
 * dimension) that a CUDA device allows for a thread block. In the CUDA Runtime API, this
 * information is available in the cudaDeviceProp structure returned by cudaGetDeviceProperties.
 * The field `maxThreadsDim` is an array of three integers (x, y, z). We simply query the
 * properties of the current device (or device 0) and print `maxThreadsDim[0]`. The program
 * should handle errors from the CUDA API calls and print a helpful message if the query fails.
 * This is a minimal standalone .cu file that compiles with `nvcc` and prints the value.
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    cudaDeviceProp prop;
    int device = 0; // Use device 0 by default

    // Get properties of the device
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error getting current device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print the maximum x-dimension of a thread block
    printf("Maximum x-dimension of a thread block on device %d: %d\n", device, prop.maxThreadsDim[0]);

    return 0;
}
