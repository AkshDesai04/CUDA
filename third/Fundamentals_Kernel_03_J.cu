/*
Launch with one block of 32 threads (one warp). Have thread 0 print a message. Then have thread 15 print a message. Then thread 31.

Thinking process:
1. We need a CUDA kernel that runs with a single block of 32 threads.
2. Each thread can use its threadIdx.x to decide if it should print.
3. We'll use printf inside the kernel for device-side output.
4. Since we want specific threads to print, we guard the printf with if (threadIdx.x == 0) etc.
5. After kernel launch, we need to synchronize the device and optionally check for errors.
6. The main function will set up the kernel launch configuration: dim3 blocks(1), dim3 threads(32).
7. We'll include <cstdio> for host-side printing and <cuda_runtime.h> for CUDA runtime.
8. We'll use nvcc to compile, and the code should be a valid .cu file.

*/

// Include necessary headers
#include <cstdio>
#include <cuda_runtime.h>

// Device kernel
__global__ void print_threads()
{
    // Get the thread index within the block
    int idx = threadIdx.x;

    // Thread 0 prints
    if (idx == 0)
    {
        printf("Thread %d: Hello from thread 0!\n", idx);
    }
    // Thread 15 prints
    if (idx == 15)
    {
        printf("Thread %d: Hello from thread 15!\n", idx);
    }
    // Thread 31 prints
    if (idx == 31)
    {
        printf("Thread %d: Hello from thread 31!\n", idx);
    }
}

int main()
{
    // Launch kernel with 1 block of 32 threads
    print_threads<<<1, 32>>>();

    // Synchronize to ensure all printf outputs are flushed
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Optional: reset device (not necessary, but good practice)
    cudaDeviceReset();

    return 0;
}
