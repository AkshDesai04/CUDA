```cuda
/*
Launch the kernel with 10 threads, but have all 10 threads try to write to `data[0]`. What value do you expect to see when you copy it back? (This is a race condition).

Thinking process:
1. The goal is to demonstrate a classic race condition in CUDA by having multiple threads concurrently write to the same memory location.
2. We will allocate a single integer on the device (`int* d_data`) and initialize it to zero.
3. The kernel will be launched with 10 threads (block size of 10). Each thread will attempt to write its own thread index (`threadIdx.x`) into `d_data[0]`.
4. Because there is no synchronization or atomic operation, the final value in `d_data[0]` after kernel execution will be nondeterministic.
5. When we copy the value back to the host and print it, the result will likely be the value written by the last thread that executed the store operation, which could be any of the values 0–9, but this is not guaranteed.
6. The program will compile with nvcc and be runnable on any CUDA-capable device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void raceKernel(int *data)
{
    // All 10 threads try to write to the same memory location
    data[0] = threadIdx.x;
}

int main(void)
{
    int h_data = 0;
    int *d_data = NULL;

    // Allocate device memory for one integer
    cudaMalloc((void**)&d_data, sizeof(int));
    cudaMemset(d_data, 0, sizeof(int));

    // Launch kernel with 10 threads in a single block
    raceKernel<<<1, 10>>>(d_data);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the resulting value
    printf("Final value in data[0] after race: %d\n", h_data);

    // Clean up
    cudaFree(d_data);

    return 0;
}
```