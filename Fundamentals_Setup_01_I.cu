/*
Aim of the program:
Query and print the PCI Bus ID and PCI Device ID for the primary GPU.

Thinking Process:
1. The requirement is to write a CUDA program in C that obtains and displays the PCI
   Bus ID and PCI Device ID for the primary GPU (device 0). 
2. CUDA Runtime API provides a structure `cudaDeviceProp` that contains
   fields `pciBusID` and `pciDeviceID`. These can be retrieved by calling
   `cudaGetDeviceProperties()`. 
3. The program should first verify that at least one GPU device is present by
   calling `cudaGetDeviceCount()`. If no device is found, print an error and exit.
4. Use error checking wrappers to simplify error handling and to provide clear
   messages if any CUDA call fails.
5. After obtaining the properties for device 0, print the PCI Bus ID and PCI Device
   ID in a human‑readable format. 
6. The program will be a single .cu file that can be compiled with `nvcc` and
   run on a system with CUDA enabled. No external dependencies are required.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* Helper macro for CUDA error checking */
#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main(void) {
    int deviceCount = 0;
    /* Get the number of CUDA-capable devices */
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable device found.\n");
        return EXIT_FAILURE;
    }

    /* Use the primary device (device 0) */
    int device = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("PCI Bus ID:    %d\n", prop.pciBusID);
    printf("PCI Device ID: %d\n", prop.pciDeviceID);

    return EXIT_SUCCESS;
}
