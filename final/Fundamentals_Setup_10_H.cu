/*
Query and print the GPU's clock rate in GHz.

Thought process:
1. Use CUDA runtime API to access device properties.
2. Call cudaGetDeviceCount to ensure at least one device exists.
3. Retrieve properties of the first device (index 0) using cudaGetDeviceProperties.
4. The clockRate field is expressed in kilohertz (kHz). To convert to gigahertz (GHz),
   divide by 1,000,000 (since 1 GHz = 1,000,000 kHz).
5. Print the clock rate in GHz with a reasonable number of decimal places.
6. Simple error checking for CUDA calls is included for robustness.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("Error getting device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-compatible GPU found.\n");
        return 0;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0); // get properties of device 0
    if (err != cudaSuccess) {
        printf("Error getting device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // clockRate is in kilohertz; convert to gigahertz
    double ghz = prop.clockRate / 1e6; // 1e6 kHz = 1 GHz
    printf("GPU Clock Rate: %.3f GHz\n", ghz);

    return 0;
}