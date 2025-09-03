```c
/*
Query and print the multiprocessor count for the default device.

Thinking:
1. Include <cuda_runtime.h> for CUDA runtime API and <stdio.h> for printing.
2. Use cudaGetDeviceCount to ensure at least one CUDA-capable device is present.
3. Pick device 0 as the default device.
4. Use cudaGetDeviceProperties to retrieve cudaDeviceProp struct for that device.
5. Print the multiProcessorCount field from the struct.
6. Add basic error checking after each CUDA API call, printing an error message and exiting if needed.
7. Keep the program minimal and self-contained, compiling with nvcc as a .cu file.
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

    int device = 0; // default device
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Device %d: %s\n", device, prop.name);
    printf("Multiprocessor count: %d\n", prop.multiProcessorCount);

    return 0;
}
```