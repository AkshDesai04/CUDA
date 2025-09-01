```cuda
/*
 * Aim of the program:
 * Extend the previous program to iterate through all available CUDA devices. For each device, use
 * `cudaGetDeviceProperties` to retrieve its properties and print its name (e.g., 'NVIDIA GeForce RTX 3080')
 * and its compute capability (e.g., 'Compute Capability: 8.6').
 *
 * Thinking and design decisions:
 * 1. The program is written in C and uses CUDA runtime API. We include <cuda_runtime.h> for the API
 *    functions and <stdio.h> for printing.
 * 2. To enumerate all devices we first call cudaGetDeviceCount. This returns the number of CUDA-capable
 *    devices present on the system.
 * 3. For each device index (0 .. count-1), we call cudaGetDeviceProperties to fill a cudaDeviceProp
 *    structure. This struct contains fields such as name, major, and minor that represent the
 *    compute capability.
 * 4. We then print the device index, the device name (prop.name), and the compute capability
 *    formatted as "major.minor".
 * 5. Basic error checking is performed after each CUDA API call; if an error occurs we print a
 *    message and exit.
 * 6. The program uses a simple macro CHECK_CUDA to wrap CUDA calls and simplify error handling.
 * 7. The code is self-contained and can be compiled with nvcc:
 *      nvcc -o device_query device_query.cu
 *    Running the resulting executable will list all CUDA devices and their compute capabilities.
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro for CUDA error checking */
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            return EXIT_FAILURE;                                    \
        }                                                           \
    } while (0)

int main(void)
{
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Found %d CUDA-capable device(s).\n", device_count);

    for (int dev = 0; dev < device_count; ++dev) {
        struct cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    }

    return EXIT_SUCCESS;
}
```