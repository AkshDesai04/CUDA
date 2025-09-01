```c
/*
Launch the printing kernel with one block of 10 threads. Observe the output.

Thinking:
- The aim is to create a simple CUDA program that demonstrates launching a kernel
  with a single block containing 10 threads. Each thread should output its own
  thread index and block index so we can observe how the CUDA runtime schedules
  threads.
- We will use the device-side printf function provided by CUDA (requires
  compute capability 2.0 or higher). The kernel will simply call printf
  with the threadIdx.x and blockIdx.x values.
- In the host code we will launch the kernel with <<<1, 10>>> configuration,
  then synchronize and check for errors. No additional logic is required.
- The program will include minimal error checking after kernel launch and
  after cudaDeviceSynchronize.
- The code is written in standard CUDA C, compileable with nvcc.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Device kernel that prints the thread and block indices
__global__ void printKernel()
{
    // Each thread prints its own threadIdx.x and blockIdx.x
    printf("Thread %d of block %d\n", threadIdx.x, blockIdx.x);
}

int main()
{
    // Launch the kernel with 1 block of 10 threads
    printKernel<<<1, 10>>>();

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Wait for GPU to finish before accessing any data
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d: %s\n",
                err, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```