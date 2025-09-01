/*
Query and print the maximum dimensions (x, y, z) of a grid of thread blocks for the primary CUDA device. This shows the maximum number of blocks you can launch in each dimension.

To solve this task we use the CUDA Runtime API. First, we include the necessary headers: <stdio.h> for printing and <cuda_runtime.h> for CUDA functions. We set the current device to 0 (the primary device) using cudaSetDevice(0). Then we declare a cudaDeviceProp structure and fill it with cudaGetDeviceProperties. The structure contains a member maxGridSize[3] which stores the maximum grid dimensions in x, y, and z. We simply print these three values using printf. Error checking is minimal for brevity; in a production setting we would check the return status of CUDA API calls. The program returns 0 on success. This is a straightforward, self-contained .cu file that compiles with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device = 0; // Primary CUDA device
    cudaError_t err;

    // Set the device (optional as 0 is default, but explicit for clarity)
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set device %d: %s\n", device, cudaGetErrorString(err));
        return -1;
    }

    // Retrieve device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n", device, cudaGetErrorString(err));
        return -1;
    }

    // Print maximum grid dimensions
    printf("Maximum grid dimensions for device %d (%s):\n", device, prop.name);
    printf("  X: %d\n", prop.maxGridSize[0]);
    printf("  Y: %d\n", prop.maxGridSize[1]);
    printf("  Z: %d\n", prop.maxGridSize[2]);

    return 0;
}
