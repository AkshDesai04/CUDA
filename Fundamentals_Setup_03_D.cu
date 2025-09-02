/*
Aim of the program: Query and print the memory bus width of the primary GPU in bits.

Thinking:
1. We need to include the CUDA runtime header <cuda_runtime.h> and standard C headers for I/O.
2. In main(), we'll call cudaGetDeviceCount to ensure at least one GPU is available.
3. We'll use device 0 as the primary GPU.
4. Retrieve cudaDeviceProp struct via cudaGetDeviceProperties.
5. The memory bus width is available in the field memoryBusWidth of cudaDeviceProp.
6. Print the value in bits using printf.
7. Add basic error checking after each CUDA API call to handle potential failures.
8. Compile with nvcc and run; it should display the bus width.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
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

    int device = 0; // primary GPU
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Memory bus width of primary GPU (device %d) in bits: %d\n", device, prop.memoryBusWidth);

    return 0;
}
