/*
Write a kernel that does an infinite loop (`while(1);`). Launch it. Observe that your program hangs. This demonstrates the asynchronous nature of kernel launches and the importance of `cudaDeviceSynchronize` for debugging.

Thought process:
- The goal is to create a minimal CUDA program that intentionally never completes so that we can observe that a kernel launch is asynchronous.
- We'll write a very small kernel that spins forever. In CUDA, a kernel that never returns will block the device thread.
- In the host code, we will launch this kernel and then immediately call cudaDeviceSynchronize() to wait for it to finish.
- Since the kernel never terminates, cudaDeviceSynchronize() will block forever, and the program will appear to hang.
- This demonstrates that kernel launches return control to the host immediately (asynchronously), and any debugging or synchronization must be explicitly requested.
- The program will print a message before launching and another after attempting to synchronize, but the second message will never be reached due to the infinite loop in the kernel.
- The code is intentionally simple and does not perform any meaningful computation.
- This example is meant purely for demonstration purposes and should not be used in production code.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void infinite_loop()
{
    while (1); // Infinite loop; never exits
}

int main()
{
    printf("Launching infinite loop kernel...\n");
    infinite_loop<<<1, 1>>>(); // Asynchronous launch

    // Wait for the kernel to finish (it never will)
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error %d: %s\n",
                err, cudaGetErrorString(err));
    } else {
        printf("Kernel finished (this line will never be printed).\n");
    }

    return 0;
}
