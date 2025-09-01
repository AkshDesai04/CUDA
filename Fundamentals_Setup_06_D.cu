/*
Print the maximum x-dimension of a grid of blocks.
In this program, we query the CUDA device properties for the current device
and print the maximum value of the x dimension for a grid of blocks.
The relevant field is maxGridSize[0] from the cudaDeviceProp struct.
We use cudaGetDeviceProperties to obtain the properties and then print
the value. Error handling is minimal: if the call fails we print an
error message and exit.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    cudaDeviceProp prop;
    int device = 0; // Using device 0; adjust if multiple GPUs exist

    // Query the device properties
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print the maximum x-dimension of a grid of blocks
    printf("Maximum x-dimension of a grid of blocks: %d\n", prop.maxGridSize[0]);

    return 0;
}
