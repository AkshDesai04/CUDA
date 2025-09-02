```cuda
/*
Aim of the program:
Write a kernel that has a 1-in-a-million chance of accessing out of bounds. Run it in a loop. Without error checking, it might run correctly 99% of the time, highlighting the danger of silent errors.

Thinking:
The goal is to demonstrate how a very rare out‑of‑bounds access can go undetected if we do not perform error checking on the device.  
I chose a simple approach: a kernel that normally writes to a small array, but if a flag is set, it writes to a far‑away location that is guaranteed to be out of bounds.  
The flag is set on the host with a 1‑in‑1,000,000 probability each iteration of a large loop.  The kernel is launched many times without checking for errors; if an out‑of‑bounds write occurs the CUDA runtime may silently ignore it or raise a warning that we do not observe, thereby mimicking silent failures.  
The program allocates a modest device array, runs the kernel many times, and finally copies back the data to confirm that normal iterations completed correctly.  No error checking after each kernel launch is performed, so any runtime errors are not reported until the very end (or not at all).  This illustrates how silent errors can slip through when error checking is omitted. 
*/

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define ARRAY_SIZE 1000
#define BLOCK_SIZE 1024  // number of threads per block

// Kernel that writes to the array normally or out of bounds based on the flag
__global__ void testKernel(int *arr, int flag)
{
    int idx = threadIdx.x;

    if (flag)
    {
        // Out-of-bounds write: target index is far beyond the allocated array
        // (assuming ARRAY_SIZE is far less than 1000000)
        arr[1000000] = idx;  // this may or may not trigger a silent error
    }
    else
    {
        // Normal write within bounds (guard against threads exceeding ARRAY_SIZE)
        if (idx < ARRAY_SIZE)
        {
            arr[idx] = idx;
        }
    }
}

int main()
{
    // Seed the random number generator
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Allocate host array
    int *h_arr = (int *)malloc(ARRAY_SIZE * sizeof(int));
    for (int i = 0; i < ARRAY_SIZE; ++i)
        h_arr[i] = 0;

    // Allocate device array
    int *d_arr = nullptr;
    cudaMalloc((void **)&d_arr, ARRAY_SIZE * sizeof(int));
    cudaMemset(d_arr, 0, ARRAY_SIZE * sizeof(int));

    // Number of kernel launches to perform
    const int NUM_LAUNCHES = 1000000;

    // Loop, launching the kernel each time
    for (int launch = 0; launch < NUM_LAUNCHES; ++launch)
    {
        // 1-in-1,000,000 chance to set flag
        int flag = (std::rand() % 1000000 == 0) ? 1 : 0;

        // Launch kernel (no error checking)
        testKernel<<<1, BLOCK_SIZE>>>(d_arr, flag);

        // Intentionally omit cudaGetLastError() and cudaDeviceSynchronize()
        // to demonstrate silent error propagation.
    }

    // Copy back results (this call synchronizes implicitly)
    cudaMemcpy(h_arr, d_arr, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Print first few values to confirm that normal operations succeeded
    printf("First 10 elements after kernel launches:\n");
    for (int i = 0; i < 10 && i < ARRAY_SIZE; ++i)
        printf("h_arr[%d] = %d\n", i, h_arr[i]);

    // Clean up
    free(h_arr);
    cudaFree(d_arr);

    return 0;
}
```