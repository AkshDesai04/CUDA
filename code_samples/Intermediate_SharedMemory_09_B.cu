/*
Aim: A 2-way bank conflict occurs if two threads in a warp access different locations in the same bank. A 32-way conflict is the worst case. Explain this.

Thinking:  
The user wants an explanation of bank conflicts, specifically 2‑way vs 32‑way conflicts. Since the developer instruction requires a CUDA program in a .cu file, I’ll provide a minimal CUDA program that demonstrates these two scenarios using shared memory.  
Shared memory in CUDA is divided into 32 banks (on current GPUs). If two or more threads in the same warp access different addresses that map to the same bank, a bank conflict occurs. A 2‑way conflict happens when exactly two threads hit the same bank; a 32‑way conflict happens when all 32 threads hit the same bank, which is the worst case because all accesses are serialized.  

The code below allocates a small shared‑memory array and uses two kernel launches:  
1. **twoWayConflictKernel** – each thread writes to an index `tid % 16`. This forces pairs of threads (e.g., 0 & 16, 1 & 17, …) to share the same bank, resulting in a 2‑way conflict.  
2. **thirtyTwoWayConflictKernel** – each thread writes to the same index `0`, causing all 32 threads to hit bank 0 and producing a 32‑way conflict.  

The kernels print which bank each thread accesses, so you can see the pattern. In a real performance test you would measure the execution time or use profiling tools, but for illustration this simple output suffices.

Below is the complete .cu file. Compile with `nvcc -arch=sm_75 bank_conflict_demo.cu -o bank_conflict_demo` and run `./bank_conflict_demo`.*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void twoWayConflictKernel(int *output)
{
    extern __shared__ int smem[];
    int tid = threadIdx.x;

    // Each thread writes to an index that causes 2-way bank conflict
    int index = tid % 16;        // indices 0..15 repeated
    smem[index] = tid;           // access shared memory

    // Determine the bank number for this index
    // Each bank holds 4 bytes (int), banks are 0..31
    int bank = index % 32;

    // Write to global memory for host output
    output[tid] = bank;
}

__global__ void thirtyTwoWayConflictKernel(int *output)
{
    extern __shared__ int smem[];
    int tid = threadIdx.x;

    // Each thread writes to the same index, forcing all to hit bank 0
    int index = 0;
    smem[index] = tid;

    int bank = index % 32;
    output[tid] = bank;
}

int main()
{
    const int warpSize = 32;
    const int numThreads = warpSize; // single warp
    const size_t size = numThreads * sizeof(int);
    int *d_output, *h_output;

    // Allocate device and host memory
    cudaMalloc((void**)&d_output, size);
    h_output = (int*)malloc(size);

    // Launch 2-way conflict kernel
    twoWayConflictKernel<<<1, numThreads, numThreads * sizeof(int)>>>(d_output);
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    printf("2-Way Conflict Kernel (bank indices per thread):\n");
    for(int i=0;i<numThreads;i++)
        printf("Thread %2d -> Bank %2d\n", i, h_output[i]);

    // Launch 32-way conflict kernel
    thirtyTwoWayConflictKernel<<<1, numThreads, numThreads * sizeof(int)>>>(d_output);
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    printf("\n32-Way Conflict Kernel (bank indices per thread):\n");
    for(int i=0;i<numThreads;i++)
        printf("Thread %2d -> Bank %2d\n", i, h_output[i]);

    // Clean up
    cudaFree(d_output);
    free(h_output);

    return 0;
}
