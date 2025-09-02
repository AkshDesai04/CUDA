```cuda
/* 
Aim of the program: Implement a single-kernel sum reduction using a grid group sync. 
A first phase has all threads reduce data into shared memory. After a block sync, 
thread 0 of each block atomically adds its partial sum to a global total. 
After a `grid.sync()`, thread 0 of block 0 reads the final total. 

Thinking process:
1. Use cooperative_groups to access `grid.sync()`. The kernel must be launched 
   cooperatively (cudaLaunchCooperativeKernel) because grid synchronization only 
   works with cooperative launch. 
2. In the kernel, load the input elements into shared memory. Perform a block‑wise 
   reduction using the standard pairwise summation technique with `__syncthreads()` 
   after each step. 
3. After the reduction, thread 0 of each block writes its partial sum into a 
   global total using `atomicAdd`. The global total resides in device global memory. 
4. All threads then perform `grid.sync()` to ensure the global total is fully 
   updated before any thread reads it. 
5. Finally, thread 0 of block 0 reads the final global total and prints it. 
   The host also copies this value back to verify correctness. 
6. Host code sets up data, allocates device memory, and launches the kernel 
   cooperatively with the appropriate shared memory size. It then retrieves 
   and prints the computed sum. 
*/

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Kernel that performs a sum reduction using grid group sync
__global__ void sumReduction(const float *input, float *global_total, int N)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory (guard against out of bounds)
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // Intra‑block reduction (tree reduction)
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 of each block writes its partial sum to global total
    if (tid == 0)
    {
        atomicAdd(global_total, sdata[0]);
    }

    // Synchronize the entire grid
    cg::grid_group grid = cg::this_grid();
    grid.sync();

    // Thread 0 of block 0 reads the final total and prints it
    if (blockIdx.x == 0 && tid == 0)
    {
        printf("Final total (inside kernel): %f\n", *global_total);
    }
}

int main()
{
    const int N = 1 << 20;           // Number of elements (1M)
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate and initialize host data
    float *h_input = new float[N];
    for (int i = 0; i < N; ++i)
        h_input[i] = 1.0f;   // simple test: sum should be N

    // Device pointers
    float *d_input = nullptr;
    float *d_total = nullptr;

    // Allocate device memory
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_total, sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize global total to zero
    cudaMemset(d_total, 0, sizeof(float));

    // Prepare kernel launch parameters
    void *kernelArgs[] = { &d_input, &d_total, &N };
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    // Launch kernel cooperatively
    cudaLaunchCooperativeKernel((void*)sumReduction,
                                dim3(blocksPerGrid), dim3(threadsPerBlock),
                                kernelArgs, sharedMemSize, 0);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    float h_total = 0.0f;
    cudaMemcpy(&h_total, d_total, sizeof(float), cudaMemcpyDeviceToHost);

    // Print result from host
    printf("Final total (host): %f\n", h_total);

    // Cleanup
    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_total);

    return 0;
}
```