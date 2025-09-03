```cuda
/*
Aim of the program: What is the size limit for constant memory? (Typically 64KB).
Thinking:
1. The CUDA programming model exposes the size of constant memory via the
   cudaDeviceProp structure (field totalConstMem). This value is a property
   of each GPU device.
2. The common limit on modern GPUs is 64 KB, but it can vary depending on
   the compute capability. We want to query this value programmatically
   so that the program reports the actual limit for the device it runs on.
3. The program will:
   - Determine the number of CUDA devices.
   - For each device, retrieve its properties with cudaGetDeviceProperties.
   - Print the device name and the constant memory size in bytes and
     kilobytes.
   - Handle any CUDA errors gracefully.
4. The code is written in CUDA C and compiles to a .cu file. It uses
   only the standard CUDA Runtime API and no external libraries.
5. The output is a simple text line per device, e.g.:
      Device 0: Quadro RTX 4000, constant memory: 65536 bytes (64.00 KB)
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Utility macro for checking CUDA errors
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            return EXIT_FAILURE;                                           \
        }                                                                  \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        size_t constMem = prop.totalConstMem; // in bytes
        double constMemKB = (double)constMem / 1024.0;

        printf("Device %d: %s, constant memory: %zu bytes (%.2f KB)\n",
               dev, prop.name, constMem, constMemKB);
    }

    return EXIT_SUCCESS;
}
```