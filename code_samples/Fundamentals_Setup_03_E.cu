/*
Query and print the maximum memory pitch in bytes allowed for memory allocations.
 
In order to satisfy this request I need to query the CUDA device properties and
extract the field that holds the maximum allowed pitch for 2‑D memory
allocations. The CUDA runtime API provides a structure
`cudaDeviceProp` that contains a member `maxPitch`.  I can obtain the
current device with `cudaGetDevice`, fill a `cudaDeviceProp` with
`cudaGetDeviceProperties`, and then print the value.  
To make the program robust I will check the return status of the CUDA
calls and report errors using `cudaGetErrorString`.  
The program will be a self‑contained `.cu` file that can be compiled with
`nvcc`.  The `printf` function from `<stdio.h>` will be used to display the
result.  No additional CUDA kernels or memory operations are required,
because the goal is purely to query and display the maximum pitch value.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Maximum memory pitch for device %d is %llu bytes.\n", dev, (unsigned long long)prop.maxPitch);

    return 0;
}
