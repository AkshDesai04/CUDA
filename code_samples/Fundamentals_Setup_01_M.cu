```c
/*
Aim of the program:
Query and print the major and minor compute capability numbers for the primary CUDA device separately.

Thinking:
- The primary CUDA device is the one currently selected by the CUDA runtime.
- We can obtain its ID using cudaGetDevice.
- The device properties, including compute capability, are available via cudaGetDeviceProperties.
- The major and minor fields of cudaDeviceProp hold the compute capability numbers.
- We'll print them using printf, with proper error checking for CUDA API calls.
- The program will be a simple console application that includes <cuda_runtime.h> and <stdio.h>.
- It will handle potential errors by printing the CUDA error string and exiting with a non-zero status.
- No additional features or command-line arguments are required.
- The code is written in C with CUDA extensions, suitable for compilation with nvcc.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting current device: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Compute Capability: Major = %d, Minor = %d\n", prop.major, prop.minor);

    return EXIT_SUCCESS;
}
```