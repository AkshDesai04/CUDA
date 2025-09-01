```cuda
/*
Aim of the program: Query and print the maximum amount of shared memory per block in bytes.
Thought process:
1. The goal is to retrieve the maximum shared memory that a CUDA device can provide to a single block.
2. Use CUDA Runtime API to query device properties.
   - cudaGetDeviceCount to get number of devices.
   - cudaGetDeviceProperties to fill a cudaDeviceProp structure.
3. The property of interest is `maxSharedMemoryPerBlock`.
4. Output the value in bytes to the console.
5. Include basic error handling: if no devices are available or if property query fails, print an error message.
6. Use `printf` for output, as it works both on host and device but here it's only host-side.
7. Keep the code minimal and self-contained, compiling with `nvcc`.
8. Place a multi-line comment at the top that repeats the aim word-for-word and includes the reasoning steps.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            return EXIT_FAILURE;                                            \
        }                                                                    \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    // Use the first device (index 0)
    int device = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("Maximum shared memory per block on device %d (%s): %zu bytes\n",
           device, prop.name, prop.maxSharedMemoryPerBlock);

    return EXIT_SUCCESS;
}
```