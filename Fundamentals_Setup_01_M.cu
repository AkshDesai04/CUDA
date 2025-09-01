```c
/*
Query and print the major and minor compute capability numbers for the primary CUDA device separately.

Thinking:
To satisfy the requirement, the program must retrieve the compute capability of the primary CUDA device and output the major and minor numbers separately. The primary CUDA device is typically device 0, but it is safer to query the current device using cudaGetDevice. Once the device ID is known, cudaGetDeviceProperties can be used to obtain a cudaDeviceProp structure, which contains the major and minor fields for compute capability. The program should include proper error checking after each CUDA runtime call and print the results to standard output. Since the task is simple, no complex kernel or CUDA streams are required. The final output will be a concise C/CUDA program that can be compiled with nvcc into a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    } while (0)

int main(void) {
    int device = 0;
    // Get the current device (primary CUDA device)
    CHECK_CUDA(cudaGetDevice(&device));

    cudaDeviceProp prop;
    // Retrieve device properties
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    // Print major and minor compute capability numbers separately
    printf("Device %d: Compute Capability %d.%d\n", device, prop.major, prop.minor);

    return EXIT_SUCCESS;
}
```