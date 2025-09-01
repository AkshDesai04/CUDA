```c
/*
 * Aim of the program:
 * Write a program to determine if the primary GPU is an integrated (sharing memory with the CPU) or a discrete device. Print the result.
 *
 * Thinking and design:
 * 1. Use the CUDA Runtime API to query the available GPU devices.
 * 2. Use cudaGetDeviceCount() to get the number of CUDA-capable devices.
 * 3. If no devices are available, print an appropriate message and exit.
 * 4. For the primary device (device ID 0), use cudaGetDeviceProperties() to retrieve a cudaDeviceProp structure.
 * 5. The cudaDeviceProp structure contains a boolean field 'integrated' which indicates whether the GPU is an integrated GPU
 *    (sharing system memory with the CPU) or a discrete GPU (has its own dedicated memory).
 * 6. Based on the value of prop.integrated:
 *    - If true, print "Integrated GPU: [device name]".
 *    - If false, print "Discrete GPU: [device name]".
 * 7. Handle any CUDA runtime errors by checking return codes and printing informative error messages.
 * 8. The program is written in C with CUDA extensions (.cu file) and can be compiled using nvcc:
 *    nvcc -o gpu_type gpu_type.cu
 * 9. The program focuses only on the primary GPU (device 0) as requested, but could be extended to iterate over all devices if needed.
 * 10. The output will be a single line indicating the type of the primary GPU and its name.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Unable to get device count: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        printf("No CUDA capable devices found.\n");
        return EXIT_SUCCESS;
    }

    int device = 0; // primary device
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Unable to get device properties for device %d: %s\n", device, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (prop.integrated) {
        printf("Integrated GPU: %s\n", prop.name);
    } else {
        printf("Discrete GPU: %s\n", prop.name);
    }

    return EXIT_SUCCESS;
}
```