/*
Query and print the total number of registers available per thread block on the primary device.

The goal of this program is to provide a simple CUDA host-side application that retrieves
and displays the amount of registers that each thread block can use on the first
CUDA-capable device found.  CUDA exposes this information in the cudaDeviceProp
structure, under the field `regsPerBlock`.  To accomplish the task we:

1. Enumerate the available CUDA devices using `cudaGetDeviceCount`.
2. Select device 0 (the primary device) and query its properties via
   `cudaGetDeviceProperties`.
3. Extract the `regsPerBlock` field from the returned `cudaDeviceProp` instance.
4. Print the value to standard output.
5. Perform basic error checking after each CUDA API call.

The program is self‑contained and compiles with any recent nvcc compiler.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = (call);                                      \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            return EXIT_FAILURE;                                      \
        }                                                              \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }

    int device = 0; // primary device
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Device %d (%s):\n", device, prop.name);
    printf("  Total registers per thread block: %d\n", prop.regsPerBlock);

    return EXIT_SUCCESS;
}
