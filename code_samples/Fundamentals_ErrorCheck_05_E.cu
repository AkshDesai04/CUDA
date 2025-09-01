```cuda
/* 
Aim: Try to get device properties for an invalid device index with `cudaGetDeviceProperties`.

Thinking:
1. The goal is to demonstrate how the CUDA runtime behaves when we request properties for a device index that doesn't exist.
2. We'll query the actual number of CUDA-capable devices on the system using `cudaGetDeviceCount`.
3. An invalid index can be chosen as the count itself, since valid indices range from 0 to count-1.
4. We'll attempt to retrieve properties for this out-of-range index and capture the returned error code.
5. The program will print the error code and its string representation to show that the request fails as expected.
6. A simple error-checking helper macro can be used for clarity, but the core logic remains straightforward.
7. No additional libraries or dependencies are required beyond the CUDA runtime header.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return EXIT_FAILURE;                              \
        }                                                     \
    } while (0)

int main(void) {
    int deviceCount = 0;
    // Get the number of CUDA devices available
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    // Choose an invalid device index: one beyond the last valid index
    int invalidIndex = deviceCount; // Since valid indices are 0 .. deviceCount-1

    // Attempt to retrieve properties for the invalid index
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, invalidIndex);

    if (err != cudaSuccess) {
        printf("Failed to get device properties for index %d: %s\n",
               invalidIndex, cudaGetErrorString(err));
    } else {
        // This block should not execute for an invalid index
        printf("Successfully retrieved properties for device %d\n", invalidIndex);
        printf("Device name: %s\n", prop.name);
    }

    return EXIT_SUCCESS;
}
```