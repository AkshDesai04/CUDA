```cpp
/*
Query and print the number of multiprocessors on the primary CUDA device (device 0).

Thinking:
1. We need to use CUDA runtime API to get device properties.
2. First, query the total number of CUDA devices to ensure at least one exists.
3. Set device 0 as the active device using cudaSetDevice(0).
4. Retrieve the properties of device 0 via cudaGetDeviceProperties().
5. The number of multiprocessors is stored in the field `multiProcessorCount`.
6. Print this number to standard output.
7. Include error handling: check return codes from CUDA API calls and print
   appropriate error messages if any call fails.
8. The program should be self‑contained in a single .cu file, compilable with
   nvcc and run on systems with CUDA installed.
*/
#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                              \
            return EXIT_FAILURE;                                            \
        }                                                                   \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    // Set device 0 as the current device
    CHECK_CUDA(cudaSetDevice(0));

    // Retrieve properties of device 0
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    // Print the number of multiprocessors
    printf("Number of multiprocessors on device 0: %d\n", prop.multiProcessorCount);

    return EXIT_SUCCESS;
}
```