/*
Aim of the program: Query and print the amount of shared memory available per thread block in bytes for the primary device.
[Thinking]
To fulfill this requirement, the program needs to:
1. Include the CUDA runtime header to access CUDA API functions.
2. Retrieve the properties of the primary CUDA device (device 0) using cudaGetDeviceProperties.
3. Extract the sharedMemPerBlock field from the cudaDeviceProp structure, which represents the amount of shared memory per block in bytes.
4. Print this value to the console.
5. Perform basic error checking after each CUDA API call to ensure graceful failure in case of problems.
The program is written in plain C and uses the CUDA runtime API, which allows it to be compiled with nvcc. The output is simple and directly prints the shared memory per block in bytes.*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device = 0; // Primary device
    cudaDeviceProp prop;
    cudaError_t err;

    // Get properties of the primary device
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print shared memory per block in bytes
    printf("Shared memory available per block on device %d: %zu bytes\n",
           device, prop.sharedMemPerBlock);

    return 0;
}