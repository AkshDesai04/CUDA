```cuda
/*
Aim of the program: Query and print the maximum z-dimension of a thread block.

Thinking:
To obtain the maximum thread block dimensions supported by a CUDA device, we use the CUDA Runtime API.
1. Include the necessary headers: <stdio.h> for printing and <cuda_runtime.h> for CUDA functions.
2. Use cudaGetDeviceCount to determine the number of CUDA-capable devices. If none are found, print an error.
3. Select device 0 (the default device) and call cudaGetDeviceProperties to fill a cudaDeviceProp structure.
4. The maximum thread block dimensions are stored in the array maxThreadsDim[3], indexed as [0] for x, [1] for y, and [2] for z.
5. Print the value of maxThreadsDim[2], which is the maximum z-dimension of a thread block.
6. Include basic error checking after CUDA API calls, using cudaGetLastError or checking the returned cudaError_t.

This program is intentionally minimal: it queries the device properties at runtime and outputs the requested value to stdout.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Utility macro for error checking
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
                    cudaGetErrorString(err));                              \
            return EXIT_FAILURE;                                            \
        }                                                                   \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    // Use the first device
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Maximum z-dimension of a thread block on device %d (%s): %d\n",
           device, prop.name, prop.maxThreadsDim[2]);

    return EXIT_SUCCESS;
}
```