```cuda
/*
Aim of the program:
Query and print the maximum x-dimension of a thread block.

Thinking process:
- The maximum x-dimension of a thread block is stored in the device property
  maxThreadsDim[0] of the cudaDeviceProp struct.
- First, we need to obtain the device count and select a device (we'll use device 0).
- Then, we call cudaGetDeviceProperties to fill a cudaDeviceProp structure.
- Finally, we print the value of maxThreadsDim[0] to stdout.
- Basic error checking is added to ensure that CUDA API calls succeed.
- The program is written in C with CUDA extensions, compiled as a .cu file.
- No external libraries beyond CUDA runtime and stdio are required.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple macro for CUDA error checking
#define CUDA_CHECK(call)                                       \
    do {                                                       \
        cudaError_t err = (call);                              \
        if (err != cudaSuccess) {                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return EXIT_FAILURE;                               \
        }                                                      \
    } while (0)

int main(void) {
    int device = 0;
    cudaDeviceProp prop;

    // Get number of devices to ensure at least one is present
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    // Set the device (optional, defaults to 0)
    CUDA_CHECK(cudaSetDevice(device));

    // Retrieve device properties
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // Print the maximum x-dimension of a thread block
    printf("Maximum x-dimension of a thread block on device %d (%s): %d\n",
           device, prop.name, prop.maxThreadsDim[0]);

    return EXIT_SUCCESS;
}
```