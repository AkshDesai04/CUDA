```cpp
/*
 * Aim of the program: Query and print the memory clock rate of the primary GPU.
 *
 * Thinking:
 * 1. The primary GPU refers to device 0 in CUDA terminology. We will use the
 *    CUDA Runtime API to query its properties.
 * 2. The `cudaDeviceProp` structure contains a field `memoryClockRate`
 *    (in kHz) which represents the memory clock rate. 
 * 3. We will call `cudaGetDeviceProperties(&prop, 0)` to fill the structure.
 * 4. After successful query, we will print the value. For readability, we
 *    convert kHz to MHz by dividing by 1000.0.
 * 5. Basic error handling is added: if any CUDA call fails, we print the
 *    error string and exit with a non-zero status.
 * 6. The program is self‑contained, requires no arguments, and produces
 *    a single line of output like: "Memory Clock Rate: 6000 MHz".
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    // Set device to primary GPU (device 0)
    int device = 0;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Retrieve device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // memoryClockRate is in kHz; convert to MHz for display
    float memoryClockMHz = prop.memoryClockRate / 1000.0f;

    printf("Memory Clock Rate: %.0f MHz\n", memoryClockMHz);

    return 0;
}
```