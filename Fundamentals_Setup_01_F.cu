/*
Query and print the amount of shared memory available per thread block in bytes for the primary device.

Thinking:
- The program needs to query CUDA device properties and print the shared memory per block.
- We'll use cudaGetDevice to obtain the current active device (default is device 0).
- Then use cudaGetDeviceProperties to retrieve a cudaDeviceProp struct for that device.
- The field prop.sharedMemPerBlock contains the amount of shared memory available per thread block in bytes.
- We'll print this value using printf.
- Error checking is added: if any CUDA API call fails, we print an error message and exit with a non-zero status.
- Include necessary headers: <stdio.h> for I/O and <cuda_runtime.h> for CUDA runtime API.
- The program will be a simple C/C++ main function that compiles with nvcc as a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0; // primary device
    cudaDeviceProp prop;
    cudaError_t err;

    // Get current device
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Get device properties
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Print the amount of shared memory per block
    printf("Shared memory per block (bytes): %zu\n", prop.sharedMemPerBlock);

    return 0;
}
