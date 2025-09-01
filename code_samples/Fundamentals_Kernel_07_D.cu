```cuda
/*
Aim of the program: 
Launch a 3D block of (4, 4, 4) threads. Have each thread print its 3D thread index (threadIdx.x, threadIdx.y, threadIdx.z).

Thinking:
- We need a kernel that runs with a 3D block configuration of 4x4x4 threads.
- Inside the kernel, each thread will call printf to output its own 3D thread indices.
- Since device printf is used, we must synchronize the device after kernel launch to ensure all output is flushed.
- We'll use the standard <stdio.h> and <cuda_runtime.h> headers.
- Basic error checking will be included to catch launch failures.
- The program will compile with nvcc and can be run directly on a CUDA-capable device.
- No other functionality is required; the program solely demonstrates 3D thread indexing and printing.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints each thread's 3D index
__global__ void print_thread_indices()
{
    printf("Thread (%d, %d, %d)\n",
           threadIdx.x,
           threadIdx.y,
           threadIdx.z);
}

int main(void)
{
    // Define a 3D block of (4, 4, 4) threads and 1 block
    dim3 threadsPerBlock(4, 4, 4);
    dim3 numBlocks(1);

    // Launch the kernel
    print_thread_indices<<<numBlocks, threadsPerBlock>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Wait for the kernel to finish and flush device printf output
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA device synchronize error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```