```cpp
/*
Aim of the program:
Extend the previous program to iterate through all available CUDA devices. For each device, use `cudaGetDeviceProperties` to retrieve its properties and print its name (e.g., 'NVIDIA GeForce RTX 3080') and its compute capability (e.g., 'Compute Capability: 8.6').

Thinking:
1. I need to enumerate all CUDA-capable devices. The CUDA Runtime API provides `cudaGetDeviceCount` for this.
2. For each device index, I can call `cudaGetDeviceProperties` to fill a `cudaDeviceProp` structure.
3. The device name is stored in `cudaDeviceProp.name` as a null-terminated string.
4. The compute capability is represented by `cudaDeviceProp.major` and `cudaDeviceProp.minor`. These are integers like 8 and 6 for compute capability 8.6.
5. I will print a line for each device in a clear format, e.g.:
   `Device 0: NVIDIA GeForce RTX 3080, Compute Capability: 8.6`
6. I'll add basic error handling using `cudaGetErrorString` to report failures of CUDA API calls.
7. The program will compile with `nvcc` and produce a single .cu file with no external dependencies.

No external libraries are required beyond the CUDA Runtime headers. The code is self‑contained and follows standard C/C++ conventions suitable for a CUDA program.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro to check CUDA API return status
#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            return EXIT_FAILURE;                                 \
        }                                                         \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Found %d CUDA device(s).\n\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, i));

        printf("Device %d: %s, Compute Capability: %d.%d\n",
               i,
               prop.name,
               prop.major,
               prop.minor);
    }

    return EXIT_SUCCESS;
}
```