/*
Write a simple __global__ function (a kernel) that does nothing. In your main function, attempt to compile this file using the NVIDIA CUDA Compiler (NVCC) command line, for example: nvcc my_program.cu -o my_program. Run the resulting executable.

Thinking:
The goal is to produce a minimal CUDA program that compiles with NVCC and runs successfully. The kernel will be empty – it performs no operations. In main we simply launch the kernel with a single block of a single thread, print a message to confirm execution, and call cudaDeviceReset() to clean up the device state. This demonstrates a complete, self‑contained CUDA program that can be compiled and executed from the command line as described in the prompt.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void emptyKernel()
{
    // Intentionally left blank; does nothing
}

int main(void)
{
    // Launch the empty kernel with 1 block and 1 thread
    emptyKernel<<<1, 1>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", err);
        return 1;
    }

    printf("Empty kernel executed successfully.\n");

    // Clean up device
    err = cudaDeviceReset();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset returned error code %d!\n", err);
        return 1;
    }

    return 0;
}
