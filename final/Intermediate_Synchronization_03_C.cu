/*
Aim: Run the program multiple times and show that the incorrect result is also non-deterministic.

Thinking:
- The goal is to demonstrate a race condition in a CUDA kernel that leads to non-deterministic results.
- We'll allocate a global array `g_arr` of size N and launch many blocks such that each thread writes to the same index `threadIdx.x`.
- Since every block writes to the same index, the final value stored at each index will be the value from the last block that wrote to it, which is nondeterministic because the order in which blocks finish execution is not defined.
- After the kernel completes, the host will sum the contents of `g_arr`. Because the stored values may differ across runs, the resulting sum will also vary, illustrating non-determinism.
- This simple example intentionally misuses CUDA by having multiple blocks write to overlapping global memory locations without any synchronization, which is a common source of nondeterministic bugs.
- The program prints the computed sum each time it runs. If you run it multiple times, you should see different sums.
*/

#include <stdio.h>
#include <cuda.h>

#define N 256          // Size of the global array
#define BLOCKS 256     // Number of blocks (will create a race)
#define THREADS 256    // Threads per block (each thread writes to same index)

// Kernel that intentionally creates a race condition by having all blocks write to the same index.
__global__ void race_kernel(int *g_arr)
{
    int idx = threadIdx.x;   // Same index for all threads in a block
    g_arr[idx] = blockIdx.x; // Each block writes its block index
}

int main(void)
{
    int *h_arr = (int *)malloc(N * sizeof(int));
    int *d_arr;

    // Allocate device memory
    cudaMalloc((void **)&d_arr, N * sizeof(int));

    // Launch the kernel: many blocks writing to same indices
    race_kernel<<<BLOCKS, THREADS>>>(d_arr);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Compute sum on host
    long long sum = 0;
    for (int i = 0; i < N; ++i)
        sum += h_arr[i];

    printf("Computed sum: %lld\n", sum);

    // Clean up
    cudaFree(d_arr);
    free(h_arr);

    return 0;
}
