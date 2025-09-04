/*
Is there a race condition if multiple threads only *read* from the same location? (No, this is safe and common).

Thinking:
- The question asks whether there is a race condition when multiple threads read from the same memory location.
- In CUDA, as in most parallel programming models, a race condition occurs when two or more threads access the same memory location concurrently and at least one access is a write, without proper synchronization.
- If all threads are only reading, there is no modification to the data, so the value read is consistent and there is no data race.
- The CUDA memory model guarantees that reads from a location that is not being written to concurrently are coherent and will return the value that was written before the reads began.
- Thus, multiple threads reading the same location is safe and common practice; e.g., broadcasting a constant to all threads or reading a shared lookup table.
- This code demonstrates that: a single value is written to device memory once, then many threads concurrently read that value and store it into a per-thread output array. No synchronization is required.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Kernel where each thread reads the same source value and writes it to its own output element */
__global__ void read_only_kernel(const int *src, int *dst, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // All threads read the same location src[0]
        int val = src[0];
        // Store the read value into the per-thread output location
        dst[idx] = val;
    }
}

int main(void)
{
    const int N = 1024;                 // Number of threads / output elements
    const int srcValue = 42;            // Value to be broadcasted
    int *h_dst = (int*)malloc(N * sizeof(int)); // Host output array

    // Allocate device memory
    int *d_src = nullptr;
    int *d_dst = nullptr;
    cudaMalloc((void**)&d_src, sizeof(int));
    cudaMalloc((void**)&d_dst, N * sizeof(int));

    // Copy the source value to device
    cudaMemcpy(d_src, &srcValue, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel: enough blocks to cover N threads
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    read_only_kernel<<<blocks, threadsPerBlock>>>(d_src, d_dst, N);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_dst, d_dst, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify that all elements are equal to srcValue
    int errors = 0;
    for (int i = 0; i < N; ++i)
    {
        if (h_dst[i] != srcValue)
        {
            printf("Mismatch at index %d: got %d, expected %d\n", i, h_dst[i], srcValue);
            errors++;
        }
    }

    if (errors == 0)
    {
        printf("All %d elements correctly read the source value %d.\n", N, srcValue);
    }
    else
    {
        printf("Found %d mismatches.\n", errors);
    }

    // Clean up
    free(h_dst);
    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}
