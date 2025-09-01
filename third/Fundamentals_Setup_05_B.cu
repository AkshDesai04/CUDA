```cpp
/*
Aim: Query and print the maximum number of registers available per block.

Thinking:
1. In CUDA, the maximum number of registers that a block can use is exposed as a device
   attribute named `cudaDevAttrMaxRegistersPerBlock`.  It can be queried with
   `cudaDeviceGetAttribute`.
2. The program will:
   - Select device 0 (or the current device).
   - Call `cudaDeviceGetAttribute` to obtain the register limit.
   - Print the value to stdout.
3. Basic error checking is performed for each CUDA API call, using a helper macro
   `CUDA_CHECK` that aborts the program with an error message if a call fails.
4. The code is self‑contained and can be compiled with `nvcc` to produce a
   single executable that prints the register limit.
5. If the device does not support this attribute (unlikely on modern GPUs),
   the program will report an error.

The result will be something like: "Maximum registers per block: 65536" (example value).

*/

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            return EXIT_FAILURE;                                    \
        }                                                           \
    } while (0)

int main(void)
{
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    int maxRegsPerBlock;
    // Query the device attribute for the maximum number of registers per block
    CUDA_CHECK(cudaDeviceGetAttribute(&maxRegsPerBlock,
                                      cudaDevAttrMaxRegistersPerBlock,
                                      device));

    printf("Maximum registers per block: %d\n", maxRegsPerBlock);

    return EXIT_SUCCESS;
}
```