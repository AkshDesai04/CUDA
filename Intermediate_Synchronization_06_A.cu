/*
 * Aim of the program (exact words from the prompt):
 * "Explain thread divergence. When a conditional is met by some threads in a warp but not others, the warp executes both paths."
 *
 * Entire thinking:
 * 
 * The goal is to produce a minimal CUDA example that demonstrates thread divergence. In CUDA, threads are grouped into warps of typically 32 threads. Within a warp, if a conditional statement leads to different execution paths for different threads, the warp has to serialize those paths: it executes one path for all threads that satisfy the condition, then the other path for the rest. This is called thread divergence.
 * 
 * To illustrate this, I will write a kernel that runs on multiple warps. Each thread will compute its global thread ID. Then, inside the kernel, I will use a simple conditional that splits the warp into two groups: even and odd thread IDs. Because even and odd IDs in a warp will be interleaved, threads within the same warp will take different branches. We will print a message from each branch so that the output clearly shows that both branches are executed for each warp. We will also store a value in an array to make the divergence effect visible from the host side.
 * 
 * The kernel will use `printf` for clarity. The host code will launch enough threads to cover at least two warps so that we can see divergence across multiple warps. Finally, the host will copy the result array back and print the values to confirm that each thread wrote the expected result.
 * 
 * This code will compile with `nvcc` and run on any GPU that supports CUDA. It demonstrates thread divergence in a very concrete way.
 */

#include <stdio.h>
#include <cuda.h>

#define THREADS_PER_BLOCK 128
#define NUM_BLOCKS 2

__global__ void divergenceKernel(int *out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Determine warp ID and lane ID
    int warpId = tid / 32;
    int laneId = tid % 32;

    // Simple conditional that causes divergence within a warp
    if (laneId % 2 == 0) {
        // Even lanes
        printf("Warp %d: Thread %d (lane %d) takes EVEN branch\n", warpId, tid, laneId);
        out[tid] = 1;
    } else {
        // Odd lanes
        printf("Warp %d: Thread %d (lane %d) takes ODD branch\n", warpId, tid, laneId);
        out[tid] = 2;
    }
}

int main()
{
    int totalThreads = THREADS_PER_BLOCK * NUM_BLOCKS;
    int *d_out;
    int *h_out = (int*)malloc(totalThreads * sizeof(int));

    cudaMalloc((void**)&d_out, totalThreads * sizeof(int));

    // Launch kernel
    divergenceKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_out);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_out, d_out, totalThreads * sizeof(int), cudaMemcpyDeviceToHost);

    // Print a summary of results
    printf("\nSummary of results:\n");
    for (int i = 0; i < totalThreads; ++i) {
        printf("Thread %d wrote %d\n", i, h_out[i]);
    }

    // Clean up
    cudaFree(d_out);
    free(h_out);

    return 0;
}
