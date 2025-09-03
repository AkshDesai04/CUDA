/*
Write a kernel that calculates the global ID and then prints whether it is even or odd.

The goal is to create a CUDA kernel that each thread can identify its own global ID
based on its block and thread indices. Once the global ID is calculated, the kernel
will determine if the ID is even or odd and print the result using device printf.
This example demonstrates basic CUDA concepts such as block and thread indexing,
global ID computation, branching on parity, and using printf from device code.
We also provide a small host program that launches the kernel with a simple
configuration (1 block of 32 threads) and checks for errors. The code is fully
self-contained and can be compiled with nvcc (e.g., nvcc -arch=sm_35 -o parity parity.cu).
*/

#include <cstdio>
#include <cstdlib>

// Kernel that prints whether each thread's global ID is even or odd
__global__ void parityKernel()
{
    // Compute the global ID of this thread
    unsigned int globalId = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine if the ID is even or odd
    bool isEven = (globalId % 2 == 0);

    // Print the result from the device
    printf("Thread %u: ID = %u is %s\n", threadIdx.x, globalId, isEven ? "even" : "odd");
}

int main()
{
    // Define number of threads per block and number of blocks
    const unsigned int threadsPerBlock = 32;
    const unsigned int numberOfBlocks = 1;

    // Launch the kernel
    parityKernel<<<numberOfBlocks, threadsPerBlock>>>();

    // Wait for GPU to finish before accessing on host
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Reset the device (optional)
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
