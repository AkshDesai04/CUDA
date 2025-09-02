/*
Aim of the program: Query and print the GPU's clock rate in GHz.

Thinking process:
1. I need to write a CUDA program that queries the GPU device properties.
2. The runtime API provides `cudaGetDeviceProperties` which returns a `cudaDeviceProp` struct.
3. In that struct, the field `clockRate` gives the core clock in kHz.
4. To convert to GHz, divide by 1,000,000 (i.e., `clockRate / 1e6`).
5. The program will:
   - Get the number of devices (optional, but good practice).
   - Choose device 0 for simplicity.
   - Retrieve its properties.
   - Compute the clock rate in GHz.
   - Print it with a reasonable number of decimal places.
6. Error handling: check return status of `cudaGetDeviceProperties` and print an error if it fails.
7. The code will be compiled as a .cu file and run on the host.
8. All standard headers and CUDA runtime header are included.
9. The main function will be straightforward and return 0 on success.
10. Since we are only querying, no kernels are launched.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;  // We'll use the first GPU device

    // Ensure we have at least one CUDA device
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return 1;
    }

    // Retrieve device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    // Convert clock rate from kHz to GHz
    float ghz = prop.clockRate / 1e6f;

    printf("GPU %d Clock Rate: %.3f GHz\n", device, ghz);

    return 0;
}
