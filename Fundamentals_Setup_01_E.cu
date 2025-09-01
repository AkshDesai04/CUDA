```cuda
/*
Aim of the program:
Query and print the memory clock rate of the primary CUDA device in kilohertz (kHz).

Thinking process:
1. The task requires us to obtain the memory clock rate from the GPU properties.
2. CUDA provides the function cudaGetDeviceProperties which fills a cudaDeviceProp
   structure with information about a device. The field memoryClockRate gives
   the memory clock rate in kilohertz (kHz).
3. We target the primary device, which is device 0. We will set the device
   explicitly with cudaSetDevice(0) to be safe.
4. After retrieving the device properties, we simply print the memoryClockRate
   using printf. No GPU kernel launch is needed.
5. We also add basic error checking: if any CUDA API call fails, we print
   an error message and exit with a non-zero status.
6. The program is self‑contained, includes the necessary headers, and can be
   compiled with nvcc to produce an executable that prints the desired value.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

int main(void) {
    int device = 0; // Primary device
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set device %d: %s\n", device, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n", device, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // memoryClockRate is already in kHz, as per CUDA documentation.
    printf("Memory clock rate of device %d: %d kHz\n", device, prop.memoryClockRate);

    return EXIT_SUCCESS;
}
```