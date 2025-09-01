/*
Aim of the program:
Query and print the `regsPerBlock` property.

Thinking:
The `regsPerBlock` field is part of the `cudaDeviceProp` structure that holds various
properties of a CUDA device. To query this value, we need to:

1. Include the CUDA runtime header (`<cuda_runtime.h>`).
2. Use `cudaGetDeviceCount` to determine how many devices are available.
3. Pick the first device (device 0) for simplicity.
4. Retrieve the device properties via `cudaGetDeviceProperties`.
5. Print the `regsPerBlock` value to standard output.

We'll add a simple error‑checking macro to wrap CUDA API calls. The program will
exit with an error message if any CUDA call fails. The output will be the integer
value of `regsPerBlock`. The code is written in C++ (but uses C‑style CUDA API)
and is suitable to compile with `nvcc` producing a `.cu` file.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Simple macro for error checking CUDA API calls
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
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

    // Print the regsPerBlock property
    printf("regsPerBlock for device %d: %d\n", device, prop.regsPerBlock);

    return EXIT_SUCCESS;
}
