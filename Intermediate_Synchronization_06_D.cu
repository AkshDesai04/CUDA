```cuda
/*
Aim of the program:
Show a valid use of `__syncthreads()` inside a conditional, where the condition is uniform across the entire block (e.g., `if (blockIdx.x == 0) { ... __syncthreads(); }`).

Thinking:
To demonstrate a correct use of `__syncthreads()` inside a conditional, the condition must be the same for all threads in a block. In this program we use the condition `blockIdx.x == 0`. When the kernel is launched with multiple blocks, only the first block (block index 0) will execute the code inside the `if`. All threads in block 0 enter the conditional and execute the barrier, while all threads in other blocks skip the entire `if` body and therefore do not hit the barrier. This satisfies CUDA’s requirement that all threads in a block either execute `__syncthreads()` or skip it.

The kernel performs a simple reduction (sum) on the first block only. The shared memory and the reduction loop are guarded by the uniform condition. This example also shows how to use shared memory, perform a reduction, and store the result back to global memory. The host code allocates data, launches the kernel, copies back the result, and prints the sum computed by block 0.

The program compiles with `nvcc` and runs on any GPU supporting CUDA. It prints the sum of the first 256 elements of an array of 1024 integers (initialized to 1), which should be 256.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

__global__ void sumFirstBlockKernel(const int *d_in, int *d_out)
{
    // Shared memory for reduction
    extern __shared__ int sdata[];

    // Only block 0 performs the sum
    if (blockIdx.x == 0)
    {
        // Load elements into shared memory
        unsigned int tid = threadIdx.x;
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        sdata[tid] = d_in[idx];
        __syncthreads();  // All threads in block 0 reach here

        // Reduction in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();  // Synchronize after each step
        }

        // Store result
        if (tid == 0)
        {
            d_out[blockIdx.x] = sdata[0];
        }
    }
    // Threads in other blocks do nothing
}

int main(void)
{
    const int N = 1024;          // Number of elements
    const int threadsPerBlock = 256;
    const int blocksPerGrid = 4; // More blocks than needed

    // Host memory allocation
    int *h_in = (int*)malloc(N * sizeof(int));
    int *h_out = (int*)malloc(blocksPerGrid * sizeof(int));

    // Initialize input data
    for (int i = 0; i < N; ++i)
    {
        h_in[i] = 1;  // All ones, so sum of first 256 elements = 256
    }

    // Device memory allocation
    int *d_in = NULL;
    int *d_out = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_in, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_out, blocksPerGrid * sizeof(int)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    size_t sharedMemSize = threadsPerBlock * sizeof(int);
    sumFirstBlockKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_in, d_out);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_out, d_out, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));

    // Print result from block 0
    printf("Sum computed by block 0: %d\n", h_out[0]);

    // Clean up
    free(h_in);
    free(h_out);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}
```