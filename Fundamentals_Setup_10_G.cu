```c
/*
Aim of the program: Query and print the `regsPerBlock` property.

Thought process:
1. The objective is to retrieve the number of registers available per block on a CUDA-capable device.
2. CUDA provides a structure `cudaDeviceProp` that holds various device attributes. The member `regsPerBlock` holds the required information.
3. Steps to implement:
   a. Include necessary headers (`cuda_runtime.h` and `stdio.h`).
   b. Define a helper macro `CHECK_CUDA` for error handling after CUDA API calls.
   c. In `main`, determine the number of CUDA devices using `cudaGetDeviceCount`.
   d. For each device (or just the first one for simplicity), use `cudaGetDeviceProperties` to fill a `cudaDeviceProp` instance.
   e. Print the `regsPerBlock` value along with the device name for clarity.
   f. Handle any errors gracefully and exit with an appropriate status.
4. The program will be self-contained, compile with `nvcc`, and produce the desired output when run on a system with CUDA drivers and hardware.

*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",       \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),            \
                    cudaGetErrorString(err));                                 \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }

    // Query properties for the first device (index 0)
    int dev = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    printf("Device %d: %s\n", dev, prop.name);
    printf("regsPerBlock: %d\n", prop.regsPerBlock);

    return EXIT_SUCCESS;
}
```