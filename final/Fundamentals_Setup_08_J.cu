```cuda
/*
Aim of the program: 
Use `cudaGetDeviceProperties` and print the `driverVersion` field from the properties struct.

Thinking process:
1. Include necessary headers: <stdio.h> for printing and <cuda_runtime.h> for CUDA runtime API.
2. Define a helper macro `CHECK_CUDA` to wrap CUDA API calls and report errors with line numbers.
3. In `main`, determine the number of CUDA-capable devices with `cudaGetDeviceCount`.
4. If at least one device exists, select device 0 and call `cudaGetDeviceProperties` to fill a `cudaDeviceProp` struct.
5. Extract the `driverVersion` field from the struct. The field is encoded as `major << 16 | minor`, so we decode it for readability.
6. Print the driver version in a human‑readable format.
7. Handle cases where no device is available or CUDA errors occur.
8. Return 0 on success.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = (call);                                  \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return EXIT_FAILURE;                                  \
        }                                                          \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable device detected.\n");
        return EXIT_FAILURE;
    }

    int device = 0; // use the first device
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    int driverVersion = prop.driverVersion;
    int major = driverVersion >> 16;
    int minor = driverVersion & 0xFFFF;

    printf("Device %d: %s\n", device, prop.name);
    printf("Driver version: %d.%d\n", major, minor);

    return EXIT_SUCCESS;
}
```