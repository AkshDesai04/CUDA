```cuda
/*
Aim of the program (word for word):
Launch with one block of 32 threads (one warp). Have thread 0 print a message. Then have thread 15 print a message. Then thread 31.

My entire thinking:
To meet the requirement we launch a single block with 32 threads. Each thread can decide whether to print based on its thread index.
Because CUDA's execution order within a warp is not guaranteed, we enforce a strict order by using __syncthreads() after each printing phase.
The sequence is:
1. thread 0 prints "Thread 0 says hello."
2. __syncthreads() ensures all threads wait until the first print is done.
3. thread 15 prints "Thread 15 says hello."
4. __syncthreads() again.
5. thread 31 prints "Thread 31 says hello."
After launching the kernel we call cudaDeviceSynchronize() on the host to wait for all GPU work to finish before the program exits.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void print_order()
{
    // Phase 1: thread 0 prints
    if (threadIdx.x == 0)
    {
        printf("Thread 0 says hello.\n");
    }
    __syncthreads(); // Ensure thread 0's print is completed before proceeding

    // Phase 2: thread 15 prints
    if (threadIdx.x == 15)
    {
        printf("Thread 15 says hello.\n");
    }
    __syncthreads(); // Ensure thread 15's print is completed before proceeding

    // Phase 3: thread 31 prints
    if (threadIdx.x == 31)
    {
        printf("Thread 31 says hello.\n");
    }
    // No need for another sync here as kernel will terminate
}

int main(void)
{
    // Launch one block with 32 threads
    print_order<<<1, 32>>>();

    // Wait for GPU to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Optional: reset device (not strictly necessary)
    cudaDeviceReset();

    return 0;
}
```