```c
/*
Aim of the program:
Query and print the computeMode of the primary GPU (e.g., Default, Exclusive, Prohibited).

Thought process:
1. The compute mode can be retrieved via the CUDA Runtime API using the attribute
   CUDA_DEVICE_ATTRIBUTE_COMPUTE_MODE, which returns an integer value representing
   the mode: 0 = Default, 1 = Exclusive, 2 = Prohibited.
2. The program will:
   a. Query the number of devices.
   b. Select the primary device (device 0).
   c. Retrieve the compute mode attribute.
   d. Translate the integer to a human-readable string.
   e. Print the result.
3. Error checking will be performed after each CUDA call to ensure that the program
   behaves correctly and reports any issues.
4. The code is written in plain C with CUDA extensions and can be compiled with
   `nvcc -o compute_mode compute_mode.cu`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return 1;
    }

    int device = 0;  // Primary GPU
    int computeMode = 0;
    err = cudaDeviceGetAttribute(&computeMode,
                                 cudaDeviceAttrComputeMode,
                                 device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get compute mode: %s\n", cudaGetErrorString(err));
        return 1;
    }

    const char *modeStr = NULL;
    switch (computeMode) {
        case cudaComputeModeDefault:
            modeStr = "Default";
            break;
        case cudaComputeModeExclusive:
            modeStr = "Exclusive";
            break;
        case cudaComputeModeProhibited:
            modeStr = "Prohibited";
            break;
        default:
            modeStr = "Unknown";
            break;
    }

    printf("Device %d compute mode: %s\n", device, modeStr);

    return 0;
}
```