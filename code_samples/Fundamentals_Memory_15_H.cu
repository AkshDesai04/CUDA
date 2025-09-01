/*
Declare a __device__ variable in a header file. Does this cause problems? (Yes, can lead to multiple definitions unless handled with extern).

When you declare a __device__ variable in a header file that is included by more than one translation unit (i.e., .cu files), each translation unit will contain a separate definition of that variable. The compiler will then produce multiple definition errors during linkage, just like with ordinary global variables. The correct way to expose a device variable defined in one source file is to place an extern declaration in the header and provide a single definition in one .cu file.

Below is a minimal CUDA program that demonstrates this pattern. It defines an extern __device__ int variable in a "header" section of the same file (for simplicity, no separate header file is created), provides a single definition of that variable, and then uses it inside a kernel. The kernel copies the device variable value to host memory for printing.

The program can be compiled with:
    nvcc -arch=sm_61 device_var_example.cu -o device_var_example
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Simulated header: device_var.h */
#ifndef DEVICE_VAR_H
#define DEVICE_VAR_H
/* Extern declaration of a __device__ variable */
extern __device__ int deviceVar;
#endif /* DEVICE_VAR_H */

/* Definition of the __device__ variable (must appear in exactly one translation unit) */
__device__ int deviceVar = 42;

/* Kernel that reads the device variable and writes its value to device memory */
__global__ void useVar(int *out) {
    out[threadIdx.x] = deviceVar;
}

int main(void) {
    /* Allocate device memory for the result */
    int *devOut;
    cudaMalloc(&devOut, sizeof(int));

    /* Launch kernel to copy deviceVar into devOut */
    useVar<<<1, 1>>>(devOut);

    /* Copy result back to host */
    int hostOut = 0;
    cudaMemcpy(&hostOut, devOut, sizeof(int), cudaMemcpyDeviceToHost);

    /* Print the value read from the device variable */
    printf("deviceVar = %d\n", hostOut);  // Expected output: 42

    /* Clean up */
    cudaFree(devOut);
    return 0;
}