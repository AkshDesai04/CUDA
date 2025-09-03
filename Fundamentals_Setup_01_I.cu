/*
Aim: Query and print the PCI Bus ID and PCI Device ID for the primary GPU.

Thinking:
To obtain the PCI identifiers for the GPU in a CUDA program, we can use the CUDA Runtime API. 
The struct cudaDeviceProp contains fields pciBusID and pciDeviceID (and pciDomainID). 
We will:
1. Choose the primary GPU (device 0).
2. Call cudaGetDeviceProperties to fill a cudaDeviceProp structure.
3. Print the pciBusID and pciDeviceID.
We include minimal error checking to ensure the calls succeed. The program is written in CUDA C (.cu file) and compiles with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device = 0; // primary GPU
    cudaDeviceProp prop;
    cudaError_t err;

    // Get device properties
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print PCI Bus ID and PCI Device ID
    printf("PCI Bus ID: %d\n", prop.pciBusID);
    printf("PCI Device ID: %d\n", prop.pciDeviceID);

    return 0;
}
