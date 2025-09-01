/*
Aim of the program:
Query the device property for `unifiedAddressing`. If true, explain what this means for host and device pointers.

Thinking process:
- The program must be a complete .cu file, written in C with CUDA extensions.
- It should query the number of CUDA-capable devices, select one (device 0), retrieve its properties via cudaGetDeviceProperties.
- The `unifiedAddressing` property indicates whether the host and device share a unified virtual address space.
- The output should state whether unified addressing is enabled and provide a concise explanation:
  * If true: the same pointer value can be used on both host and device, eliminating explicit copies.
  * If false: the host and device have separate address spaces; memory must be explicitly copied between them.
- Basic error checking is included to handle CUDA API calls gracefully.
- The code uses standard CUDA headers and prints results to stdout.

The code below follows these requirements and is ready to compile with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Macro for error checking */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            return EXIT_FAILURE;                                                \
        }                                                                       \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    /* Use device 0 for demonstration */
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Device %d (%s):\n", device, prop.name);
    printf("  Unified Addressing: %s\n", prop.unifiedAddressing ? "True" : "False");

    if (prop.unifiedAddressing) {
        printf("\nUnified addressing is enabled on this device.\n");
        printf("This means that host and device pointers refer to the same\n");
        printf("virtual memory address. A single pointer can be used to access\n");
        printf("memory both from the host and from kernels, eliminating the need\n");
        printf("to explicitly copy data between host and device memory.\n");
    } else {
        printf("\nUnified addressing is not enabled on this device.\n");
        printf("Host and device have separate virtual address spaces. Memory\n");
        printf("must be explicitly copied between host and device using\n");
        printf("cudaMemcpy or similar functions.\n");
    }

    return EXIT_SUCCESS;
}
