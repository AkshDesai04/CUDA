```cuda
/* 
Aim of the program: 
"Launch a normal, fast kernel, followed immediately by the hanging kernel. What is the program's behavior?"

Thinking: 
The program demonstrates how CUDA streams and kernel launches interact when one kernel stalls. 
We will define two kernels:

1. fastKernel: A simple, quick kernel that writes a value to global memory or does a trivial calculation.
2. hangingKernel: A kernel that enters an infinite loop, effectively hanging.

In main, we will launch fastKernel and immediately launch hangingKernel on the default stream (stream 0). 
Because they share the same stream, the second kernel will not start executing until the first one has finished. 
After the first kernel completes, the second kernel starts executing and never terminates. 
The host program will block at the implicit cudaDeviceSynchronize that occurs when the program ends or when we explicitly call cudaDeviceSynchronize, because the hanging kernel never returns. 
We will also print messages from the host before and after each kernel launch to observe the program's behavior. 
If we were to add an explicit cudaDeviceSynchronize after launching the hanging kernel, it would never return, demonstrating the hanging behavior. 
No other streams or synchronization primitives are used, so the behavior is straightforward: the program will terminate the first kernel, start the second, and then hang indefinitely. 
This showcases the importance of ensuring kernels terminate and the impact on program flow.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Fast kernel: does a trivial computation
__global__ void fastKernel(int *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 2;  // simple operation
    }
}

// Hanging kernel: infinite loop
__global__ void hangingKernel(int *dummy)
{
    // Do nothing useful; just loop forever
    while (true) {
        // Prevent compiler optimization
        *dummy = *dummy;
    }
}

int main()
{
    const int N = 1024;
    size_t size = N * sizeof(int);

    int *d_data;
    int *d_dummy;

    // Allocate device memory
    cudaMalloc((void**)&d_data, size);
    cudaMalloc((void**)&d_dummy, sizeof(int));
    cudaMemset(d_dummy, 0, sizeof(int));

    printf("Launching fastKernel...\n");
    fastKernel<<<(N + 255)/256, 256>>>(d_data, N);

    printf("Launching hangingKernel immediately after fastKernel...\n");
    hangingKernel<<<1, 1>>>(d_dummy);

    // Wait for all kernels to finish (this will block indefinitely due to hangingKernel)
    cudaDeviceSynchronize();

    // This point will never be reached because hangingKernel never terminates
    printf("Program has finished execution.\n");

    // Cleanup (unreachable in this example, but good practice)
    cudaFree(d_data);
    cudaFree(d_dummy);

    return 0;
}
```