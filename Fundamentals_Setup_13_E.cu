/*
Aim: Query and print the device's multiprocessor count.

Thinking:
The goal is to write a minimal CUDA C program that retrieves and displays the number of streaming multiprocessors (SMs) on the GPU device. 
We will:
1. Include <cuda_runtime.h> and <stdio.h>.
2. In main, we call cudaGetDeviceCount to check how many devices are available. If none, print an error and exit.
3. For simplicity, we will query the first device (device 0). 
4. Use cudaGetDeviceProperties to fill a cudaDeviceProp struct, which contains the field multiProcessorCount.
5. Print the device name and the multiprocessor count.
6. Handle errors from CUDA API calls using cudaGetErrorString for debugging.
7. Return 0 on success.

This satisfies the requirement: a single .cu file, no external dependencies, outputs the multiprocessor count.

*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount returned error %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }

    int device = 0; // We'll query the first device
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties returned error %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Device %d: %s\n", device, prop.name);
    printf("Multiprocessor Count: %d\n", prop.multiProcessorCount);

    return EXIT_SUCCESS;
}
