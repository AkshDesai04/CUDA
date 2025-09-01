/*
Aim: Launch the empty kernel with a configuration of 1 block and 1 thread.

Thinking:
- The goal is to create a minimal CUDA program that defines an empty kernel
  (i.e. a kernel that performs no operations) and launches it with a
  configuration consisting of a single block and a single thread.
- The program will include the necessary headers, define the kernel
  `emptyKernel`, and in `main` will launch it with the `<<<1, 1>>>` launch
  configuration.
- After the launch, the program will synchronize the device and check for
  any launch errors, printing a simple message indicating that the kernel
  completed successfully.
- Since the kernel does nothing, there is no need for any data transfer
  or device memory allocation.
- The code will compile with nvcc and run on any CUDA-capable device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Empty kernel that does nothing
__global__ void emptyKernel()
{
    // No operation
}

int main(void)
{
    // Launch the empty kernel with 1 block and 1 thread
    emptyKernel<<<1, 1>>>();

    // Wait for the kernel to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Empty kernel launched with 1 block and 1 thread successfully.\n");
    return EXIT_SUCCESS;
}
