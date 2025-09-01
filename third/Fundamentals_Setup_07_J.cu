```cuda
/*
 * Aim of the program:
 * Check if the device is part of a TCC (Tesla Compute Cluster) (isTCCDriver).
 *
 * Entire thinking:
 * 1. The CUDA Runtime API provides a way to query various device attributes using
 *    cudaDeviceGetAttribute. One of these attributes is cudaDevAttrIsTCCEnabled,
 *    which returns whether a device is operating in TCC mode (returns 1 if
 *    TCC-enabled, 0 otherwise).
 * 2. The program should:
 *    - Get the total number of CUDA-capable devices with cudaGetDeviceCount.
 *    - For each device, retrieve its properties (name, memory, etc.) using
 *      cudaGetDeviceProperties for a more informative output.
 *    - Query the cudaDevAttrIsTCCEnabled attribute for that device.
 *    - Print out whether the device is TCC-enabled or not.
 * 3. Error handling: define a macro CUDA_CHECK to wrap runtime calls and
 *    print the error message with cudaGetErrorString if a call fails.
 * 4. The program is written entirely in CUDA C (.cu file) and uses only the
 *    runtime API, so it can be compiled with `nvcc`. It does not depend on
 *    any other files or libraries.
 * 5. The code will compile and run on any system with a CUDA-capable GPU.
 * 6. The output will look like:
 *      Device 0: Tesla V100-SXM2-32GB, TotalGlobalMem = 34359738368 bytes, TCC Enabled = 1
 *    or
 *      Device 0: GeForce RTX 3090, TotalGlobalMem = 17179869184 bytes, TCC Enabled = 0
 *
 * 7. The program is kept simple and self-contained.
 */

#include <cuda_runtime.h>
#include <stdio.h>

/* Macro for checking CUDA runtime API errors */
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            return EXIT_FAILURE;                                             \
        }                                                                     \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        int isTCCEnabled = 0;
        /* Query the TCC enabled attribute */
        CUDA_CHECK(cudaDeviceGetAttribute(&isTCCEnabled,
                                          cudaDevAttrIsTCCEnabled,
                                          dev));

        printf("Device %d: %s, TotalGlobalMem = %llu bytes, TCC Enabled = %d\n",
               dev,
               prop.name,
               (unsigned long long)prop.totalGlobalMem,
               isTCCEnabled);
    }

    return EXIT_SUCCESS;
}
```