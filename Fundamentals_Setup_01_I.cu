/*
Aim: Query and print the PCI Bus ID and PCI Device ID for the primary GPU.

Thinking:
To achieve this, we use the CUDA Runtime API. The primary GPU is typically device 0, so we set `int device = 0;`. We then call `cudaGetDeviceProperties` to fill a `cudaDeviceProp` structure with information about the GPU. This structure contains the `pciBusID` and `pciDeviceID` fields that we want to print. After retrieving the properties, we output these IDs using `printf`. We also perform basic error checking: if `cudaGetDeviceProperties` returns an error, we print the error string and exit with a non-zero status. This simple program is self-contained and should compile with `nvcc` and run on any system with CUDA installed, displaying the requested PCI identifiers for the first GPU in the system. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;  // primary GPU
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("PCI Bus ID: %d\nPCI Device ID: %d\n", prop.pciBusID, prop.pciDeviceID);
    return 0;
}
