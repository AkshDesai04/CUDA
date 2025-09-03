/*
Aim of the program:
Query and print the maximum x-dimension of a thread block.

Thinking process:
1. The program needs to retrieve device properties via cudaGetDeviceProperties.
2. The maximum x-dimension of a thread block is stored in the field `maxThreadsDim[0]` of the cudaDeviceProp structure.
3. We will use `cudaGetDeviceProperties(&prop, 0)` to get properties for device 0 (the default device).
4. Error handling: If cudaGetDeviceProperties fails, print an error message and exit.
5. On success, print the value of `prop.maxThreadsDim[0]` in a readable format.
6. The program is written in C with CUDA runtime API, compiled as a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Maximum x-dimension of a thread block: %d\n", prop.maxThreadsDim[0]);

    return 0;
}
