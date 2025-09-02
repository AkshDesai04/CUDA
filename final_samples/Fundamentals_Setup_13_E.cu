```c
/*
Aim of the program:
Query and print the device's multiprocessor count.

Thinking process:
The task is to create a CUDA C program that queries the number of multiprocessors
(also known as SMs) on the current GPU device and prints this value to standard
output. The relevant CUDA API for retrieving device properties is
cudaGetDeviceProperties, which fills a cudaDeviceProp structure. The field
`multiProcessorCount` holds the desired count. We'll set the device to 0
by default (the first device), but we could optionally let the user specify
another device via a command-line argument. For simplicity, we'll just use
device 0. The program will:
1. Include <stdio.h> for printf.
2. Include <cuda_runtime.h> for CUDA runtime functions.
3. In main, call cudaGetDeviceProperties on device 0.
4. If the call fails, print an error message and exit with non-zero status.
5. Otherwise, print the number of multiprocessors.
The program is self‑contained, compilable with nvcc, and requires no
additional files.*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0; // default to first GPU
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    printf("Device %d has %d multiprocessor(s).\n",
           device, prop.multiProcessorCount);

    return 0;
}
```