```cpp
/* 
Extend the previous program to iterate through all available CUDA devices. For each device, use `cudaGetDeviceProperties` to retrieve its properties and print its name (e.g., 'NVIDIA GeForce RTX 3080') and its compute capability (e.g., 'Compute Capability: 8.6'). 

My thinking process for this task is as follows:

1. The user wants a program that enumerates all CUDA devices present on the system. 
2. The CUDA Runtime API provides `cudaGetDeviceCount` to obtain the number of devices, and `cudaGetDeviceProperties` to get detailed information about a specific device.
3. We need to include the necessary headers: `<cstdio>` for printing and `<cuda_runtime.h>` for CUDA runtime functions.
4. For error handling, I'll create a helper macro `CUDA_CHECK` that checks the return value of a CUDA call and prints an error message if it fails.
5. The program will:
   - Call `cudaGetDeviceCount` to determine how many devices are available.
   - If the count is zero, print a message and exit.
   - Otherwise, loop over each device index:
     * Optionally set the current device with `cudaSetDevice` (not strictly required for property queries, but keeps the logic clear).
     * Retrieve the properties with `cudaGetDeviceProperties`.
     * Print the device name and compute capability (major.minor).
6. After printing all device information, the program will exit successfully.
7. Since the requirement is to produce a `.cu` file, I'll write the code in a single file, with the multi‑line comment at the top containing the exact prompt text and my reasoning.
8. No other supporting files or text will be output, only the `.cu` content.

Now I'll present the complete CUDA source file below.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s\n",  \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            return EXIT_FAILURE;                                         \
        }                                                                \
    } while (0)

int main(void)
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Found %d CUDA device(s):\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        // Optional: set the device, though not strictly necessary for properties
        CUDA_CHECK(cudaSetDevice(dev));

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    }

    return EXIT_SUCCESS;
}
```