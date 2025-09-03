/*
Aim of the program:
Query and print the maximum dimensions (x, y, z) of a grid of thread blocks for the primary CUDA device. 
This shows the maximum number of blocks you can launch in each dimension.

Thinking process:
1. Include the necessary headers: <stdio.h> for printing and <cuda_runtime.h> for CUDA API calls.
2. Retrieve the current CUDA device using cudaGetDevice. The primary device is typically device 0 unless changed.
3. Get the device properties with cudaGetDeviceProperties, which returns a cudaDeviceProp structure.
4. The structure contains the maximum grid size in each dimension as an array maxGridSize[3].
5. Print these values in a readable format.
6. Add minimal error checking for the CUDA calls to ensure the program reports issues if the device is unavailable.
7. Keep the program simple, self-contained, and compile-able with nvcc as a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0; // primary device
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting current CUDA device: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("CUDA Device %d: %s\n", device, prop.name);
    printf("Maximum grid dimensions (blocks): (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    return EXIT_SUCCESS;
}
