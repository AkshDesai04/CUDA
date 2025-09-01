/*
Aim of the program: Query and print the memory clock rate of the primary GPU.

Thinking process:
1. The requirement is to write a CUDA C program that queries the memory clock rate
   of the primary GPU (device 0). 
2. In CUDA, device properties are obtained via cudaGetDeviceProperties which
   fills a cudaDeviceProp structure. The memory clock rate is stored in the 
   `memoryClockRate` field, expressed in kHz.
3. We'll use cudaGetDeviceCount to confirm at least one GPU is present, then
   set the active device to 0. 
4. Retrieve the device properties, then convert the memory clock rate from
   kHz to MHz for a more readable output (divide by 1000.0).
5. Print the result with a clear message. 
6. Include basic error checking: if cudaGetDeviceCount fails or if the
   device count is zero, report an error.
7. Use `cudaGetLastError` to catch any runtime errors from CUDA API calls.
8. Keep the code self-contained and minimal, including only necessary headers.
9. Ensure the file is a .cu file; no additional supporting files or text are
   output, only the code itself.

The final program below follows these steps and prints the memory clock rate
in MHz with an appropriate message.
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
        fprintf(stderr, "No CUDA-capable device found.\n");
        return 1;
    }

    // Use the primary device (device 0)
    int device = 0;
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    // memoryClockRate is in kHz; convert to MHz
    double memClockMHz = prop.memoryClockRate / 1000.0;

    printf("Memory clock rate of primary GPU (device %d): %.2f MHz\n", device, memClockMHz);

    // Optional: check for any pending errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
