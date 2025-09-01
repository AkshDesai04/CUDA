/*
Aim of the program:
Query and print the multiprocessor count for the default device.

Thought process:
1. Include necessary headers: <cuda_runtime.h> for CUDA runtime API and <stdio.h> for printing.
2. Retrieve the default device ID using cudaGetDevice().
3. Allocate a cudaDeviceProp structure to hold device properties.
4. Call cudaGetDeviceProperties() with the device ID to fill the structure.
5. Extract the multiProcessorCount field from the structure.
6. Print the count using printf().
7. Include simple error checking for each CUDA call to handle potential failures.
8. Compile as a .cu file and run with nvcc; the program will output the multiprocessor count of the default CUDA device.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Number of multiprocessors on device %d: %d\n", device, prop.multiProcessorCount);

    return 0;
}
