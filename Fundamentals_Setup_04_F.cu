```cpp
/*
Aim: Query and print the `computeMode` of the primary GPU (e.g., Default, Exclusive, Prohibited).

Thinking:
1. The CUDA runtime API provides `cudaGetDeviceProperties` to retrieve properties of a device, including the `computeMode` field.
2. `cudaDeviceProp::computeMode` is an enum of type `cudaComputeMode`, with possible values:
   - `cudaComputeModeDefault` (0)
   - `cudaComputeModeExclusiveProcess` (1)
   - `cudaComputeModeExclusiveThread` (2)
   - `cudaComputeModeProhibited` (3)
3. We will query the first device (device 0) which is typically the primary GPU.
4. Use a helper macro `CHECK_CUDA_CALL` to handle errors from CUDA API calls gracefully.
5. Map the enum value to a human-readable string and print it to standard output.
6. The program is self‑contained and can be compiled with `nvcc` as a `.cu` file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CHECK_CUDA_CALL(call)                                         \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));    \
            return EXIT_FAILURE;                                     \
        }                                                             \
    } while (0)

// Function to convert compute mode enum to string
const char* computeModeToString(cudaComputeMode mode) {
    switch (mode) {
        case cudaComputeModeDefault:
            return "Default";
        case cudaComputeModeExclusiveProcess:
            return "Exclusive Process";
        case cudaComputeModeExclusiveThread:
            return "Exclusive Thread";
        case cudaComputeModeProhibited:
            return "Prohibited";
        default:
            return "Unknown";
    }
}

int main(void) {
    int device = 0; // primary GPU
    // Ensure the device exists
    int deviceCount = 0;
    CHECK_CUDA_CALL(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable device found.\n");
        return EXIT_FAILURE;
    }
    if (device >= deviceCount) {
        fprintf(stderr, "Device index %d out of range. Only %d device(s) available.\n",
                device, deviceCount);
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    CHECK_CUDA_CALL(cudaGetDeviceProperties(&prop, device));

    const char* modeStr = computeModeToString(prop.computeMode);
    printf("Compute mode of primary GPU (device %d) is: %s\n", device, modeStr);

    return EXIT_SUCCESS;
}
```