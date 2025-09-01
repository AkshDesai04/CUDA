```cpp
/* 
Query and print the maximum z-dimension of a thread block. 
This program uses the CUDA Runtime API to obtain the device properties for the default GPU (device 0). 
It retrieves the maximum number of threads allowed per block in each dimension, specifically the z dimension, 
using the `maxThreadsDim[2]` field of the `cudaDeviceProp` structure. 
The result is printed to standard output. 
The program performs basic error checking on CUDA API calls and reports any failures to the user. 
The aim is to demonstrate how to query hardware limits programmatically in CUDA. 
*/

#include <cstdio>
#include <cuda_runtime.h>

// Helper macro for checking CUDA errors
#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",  \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return EXIT_FAILURE;                                  \
        }                                                          \
    } while (0)

int main() {
    int device = 0;
    cudaDeviceProp prop;

    // Get properties for device 0
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    // The maximum number of threads per block in the z-dimension
    int maxZThreads = prop.maxThreadsDim[2];

    printf("Maximum z-dimension of a thread block: %d\n", maxZThreads);

    return EXIT_SUCCESS;
}
```