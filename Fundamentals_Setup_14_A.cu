```cuda
/*
Aim of the program:
A more direct property is `deviceOverlap`. Query and print this boolean property for the primary device.

Thought process:
- The property `deviceOverlap` is part of the `cudaDeviceProp` structure, which holds information about a CUDA device.
- To obtain this property, we need to:
  1. Determine the primary device. In CUDA, the "primary" device is usually the device that is currently set as the current device for the calling host thread. We can retrieve it with `cudaGetDevice()`.
  2. Use `cudaGetDeviceProperties()` to fill a `cudaDeviceProp` structure for that device ID.
  3. Access the `deviceOverlap` field of the structure. It is an `int` but represents a boolean (0 or 1).
- For robustness, we will check for CUDA errors after each API call.
- The program will simply print the value of `deviceOverlap` to `stdout`.

Implementation details:
- Include <cuda_runtime.h> for CUDA API declarations.
- Use a helper function `checkCudaErrors` to wrap CUDA calls and exit on error.
- In `main()`, call `cudaGetDevice` to get the current device ID.
- Allocate a `cudaDeviceProp` struct and fill it.
- Print the property as a boolean (True/False) for readability.

The code is self‑contained and can be compiled with `nvcc` as a standard .cu file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CHECK_CUDA(call)                                               \
    do {                                                               \
        cudaError_t err = (call);                                      \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

int main(void) {
    int device_id = 0;
    // Get the current device (primary device)
    CHECK_CUDA(cudaGetDevice(&device_id));

    // Retrieve device properties
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));

    // Print the deviceOverlap property
    printf("Device %d ('%s') deviceOverlap: %s\n",
           device_id,
           prop.name,
           prop.deviceOverlap ? "True" : "False");

    return 0;
}
```