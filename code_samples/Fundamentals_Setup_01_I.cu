/*
Aim: Query and print the PCI Bus ID and PCI Device ID for the primary GPU.

Thinking:
- The program uses the CUDA Runtime API to query the properties of the primary GPU (device 0).
- It calls cudaGetDeviceCount to ensure at least one GPU is present.
- Then it calls cudaGetDeviceProperties to retrieve a cudaDeviceProp structure for device 0.
- The pciBusID and pciDeviceID fields of this structure hold the desired identifiers.
- The program prints these IDs in a readable format.
- Error handling is included: any CUDA call failure is reported with the CUDA error string and the program exits with a non-zero status.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices detected.\n");
        return 1;
    }

    int primaryDevice = 0; // primary GPU
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, primaryDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Primary GPU (device %d):\n", primaryDevice);
    printf("  PCI Bus ID   : %d\n", prop.pciBusID);
    printf("  PCI Device ID: %d\n", prop.pciDeviceID);

    return 0;
}
