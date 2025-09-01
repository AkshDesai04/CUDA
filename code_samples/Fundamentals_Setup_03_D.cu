/*
Aim of the program:
Query and print the memory bus width of the primary GPU in bits.

Thinking:
The CUDA runtime API provides a struct `cudaDeviceProp` that contains detailed
information about each CUDA device. One of its fields, `memoryBusWidth`, reports
the width of the device's memory bus in bits. To obtain this value we simply
need to:

1. Select the primary device (device 0, which is the default when no
   other device has been chosen).
2. Call `cudaGetDeviceProperties` to fill a `cudaDeviceProp` instance.
3. Access the `memoryBusWidth` field and print it.

Error handling is important: each CUDA API call returns a
`cudaError_t`.  A small helper macro is used to check the return status and
report any failure.  If the device does not report a bus width (unlikely for
modern GPUs), the code will still print the value (typically 0).

The program is self‑contained, compiles with `nvcc`, and uses only the
CUDA runtime API and standard C++ I/O.
*/

#include <iostream>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " - " << cudaGetErrorString(err) << std::endl;      \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

int main()
{
    // Assume primary device (device 0)
    int device = 0;
    cudaDeviceProp prop;

    // Retrieve device properties
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    // Print the memory bus width in bits
    std::cout << "Memory bus width of GPU " << device
              << " : " << prop.memoryBusWidth << " bits" << std::endl;

    return 0;
}
