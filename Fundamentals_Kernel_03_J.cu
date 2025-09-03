/*
Aim: Launch with one block of 32 threads (one warp). Have thread 0 print a message. Then have thread 15 print a message. Then thread 31.

Thoughts:
- The program will use a simple CUDA kernel that launches one block with 32 threads.
- Each thread will check its threadIdx.x and, if it matches 0, 15, or 31, it will print a message using device printf.
- To ensure the messages appear in the required order (0 first, then 15, then 31), the kernel will use __syncthreads() barriers between each printing section.
- Device printf requires a GPU of compute capability 2.0 or higher; the code is written accordingly.
- The host code will launch the kernel, then call cudaDeviceSynchronize() to ensure all device output is flushed before the program exits.
- Minimal error checking is included for clarity.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void printKernel()
{
    // Thread 0 prints
    if (threadIdx.x == 0) {
        printf("Thread %d: Hello from thread 0!\n", threadIdx.x);
    }
    __syncthreads(); // Ensure thread 0 has finished printing before next thread

    // Thread 15 prints
    if (threadIdx.x == 15) {
        printf("Thread %d: Hello from thread 15!\n", threadIdx.x);
    }
    __syncthreads(); // Ensure thread 15 has finished printing before next thread

    // Thread 31 prints
    if (threadIdx.x == 31) {
        printf("Thread %d: Hello from thread 31!\n", threadIdx.x);
    }
}

int main()
{
    // Launch configuration: 1 block, 32 threads (one warp)
    dim3 blockDim(32, 1, 1);
    dim3 gridDim(1, 1, 1);

    // Launch the kernel
    printKernel<<<gridDim, blockDim>>>();

    // Wait for GPU to finish before accessing on host
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
