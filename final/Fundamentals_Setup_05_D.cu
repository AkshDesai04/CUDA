```cpp
/*
Query and print the maximum amount of shared memory per block in bytes.

I need to write a simple CUDA C program that queries the GPU device
properties and prints the maximum shared memory that can be allocated
per block, which is given by the field `sharedMemPerBlock` in the
`cudaDeviceProp` structure. The program will:

1. Include the necessary headers (`stdio.h` and `cuda_runtime.h`).
2. Use `cudaGetDevice` to get the current device (or simply use device 0).
3. Call `cudaGetDeviceProperties(&prop, device)` to fill a
   `cudaDeviceProp` struct.
4. Print `prop.sharedMemPerBlock` in bytes.
5. Handle any CUDA errors in a simple macro for clarity.

The code is minimal, self‑contained, and compiles as a `.cu` file.

*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro to check CUDA API calls
#define CHECK_CUDA(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            return EXIT_FAILURE;                                           \
        }                                                                   \
    } while (0)

int main(void) {
    int device = 0;
    cudaDeviceProp prop;

    // Get the properties of the device
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    // Print the maximum shared memory per block in bytes
    printf("Maximum shared memory per block: %zu bytes\n",
           (size_t)prop.sharedMemPerBlock);

    return EXIT_SUCCESS;
}
```