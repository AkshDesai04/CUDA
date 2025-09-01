/*
Query and print the maximum amount of shared memory per block in bytes.
This program aims to retrieve the hardware limit on shared memory that can be allocated per block on the current CUDA device. 
To accomplish this, we use the CUDA Runtime API function cudaGetDeviceProperties, which fills a cudaDeviceProp structure with various device attributes, including sharedMemPerBlock. 
The program first selects the default device (device 0) and obtains its properties. It then prints the value of sharedMemPerBlock, which is reported in bytes. 
Error checking is performed after each CUDA API call to ensure robust execution. If the program fails to retrieve the properties, it will output an error message and exit with a non-zero status. 
The output of the program will be a single line indicating the maximum shared memory per block for the selected device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0; // default device
    cudaDeviceProp prop;
    cudaError_t err;

    // Get device properties
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Maximum shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);

    return 0;
}
