/*
Print the maximum y-dimension of a grid of blocks.

I decided to write a straightforward CUDA C program that queries the
device properties using cudaGetDeviceProperties. The maximum grid
dimension is stored in the maxGridSize array inside the
cudaDeviceProp structure. Since we are interested in the y‑dimension,
we access index 1 (0‑based indexing: x is 0, y is 1, z is 2).
The program sets the device to 0, retrieves the properties, and
prints the value. Basic error handling is added to ensure the call
succeeds before accessing the field. This simple approach is
sufficient for the task of reporting the maximum y‑dimension of a
grid of blocks.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0; // use default device
    cudaDeviceProp prop;

    // Query device properties
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    // Print the maximum y-dimension of a grid of blocks
    printf("Maximum grid y-dimension for device %d: %d\n",
           device, prop.maxGridSize[1]);

    return 0;
}
