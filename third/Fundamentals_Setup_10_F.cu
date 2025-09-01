```c
/*
Aim of the program: Query and print the memory bus width in bits.

Thinking process:
1. To determine the memory bus width, we use the CUDA Runtime API function
   cudaGetDeviceProperties which fills a cudaDeviceProp structure.
   The structure contains a field called memoryBusWidth (in bits) that
   represents the width of the memory bus of the device.
2. The program should:
   a. Include necessary headers (<cuda_runtime.h>, <stdio.h>).
   b. Define a simple error-checking macro to handle CUDA API return codes.
   c. Retrieve the number of devices with cudaGetDeviceCount.
   d. For each device (or just device 0 for simplicity), call
      cudaGetDeviceProperties to obtain its properties.
   e. Print the memoryBusWidth field in bits.
3. The code is written in plain C with CUDA extensions, so the file is
   a .cu file and can be compiled with nvcc. No additional dependencies
   or support files are required.

The program is intentionally minimal and self‑contained, focusing only on
the requested query and output.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for error checking CUDA API calls
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = (call);                                      \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            return EXIT_FAILURE;                                       \
        }                                                              \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    // For simplicity, query the first device (device 0)
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("Device %d: %s\n", dev, prop.name);
    printf("Memory bus width: %d bits\n", prop.memoryBusWidth);

    return EXIT_SUCCESS;
}
```