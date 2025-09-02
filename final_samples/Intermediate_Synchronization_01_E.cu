```cpp
/*
Aim of the program:
Combine the approaches: a single kernel computes block-level partial sums, 
and then thread 0 of each block uses `atomicAdd` on a single global counter (*d_final_sum) 
to add its result.

Thinking:
- Allocate an input array of integers on the host and fill it with some values.
- Allocate corresponding device memory for the input array and the global sum.
- The kernel will:
  * Load the input values into shared memory.
  * Perform a reduction within each block to compute the block's partial sum.
  * Thread 0 of each block will use atomicAdd to add its block partial sum to the
    global sum stored in *d_final_sum.
- After the kernel finishes, copy *d_final_sum back to the host and print the result.
- Include error checking for CUDA API calls and kernel launches.
- Use a simple block size (e.g., 256 threads) and calculate grid size accordingly.
- Use float or int? Here we'll use int for simplicity.
- Provide comments for clarity and maintainability.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                   \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Kernel: each block computes its partial sum, then thread 0 adds it to global sum
__global__ void sumKernel(const int* __restrict__ d_in, int n, int* d_global_sum)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    // Load elements into shared memory, zero if out-of-range
    sdata[tid] = (idx < n) ? d_in[idx] : 0;
    __syncthreads();

    // Reduce within block
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes block's partial sum to global sum via atomicAdd
    if (tid == 0) {
        atomicAdd(d_global_sum, sdata[0]);
    }
}

int main()
{
    const int N = 1 << 20; // 1M elements
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate and initialize host memory
    int* h_in = (int*)malloc(N * sizeof(int));
    if (!h_in) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        h_in[i] = 1; // simple test: sum should be N
    }

    // Allocate device memory
    int* d_in = nullptr;
    int* d_global_sum = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_in, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_global_sum, sizeof(int)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));
    // Initialize global sum to zero
    int zero = 0;
    CHECK_CUDA(cudaMemcpy(d_global_sum, &zero, sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    size_t sharedMemSize = BLOCK_SIZE * sizeof(int);
    sumKernel<<<GRID_SIZE, BLOCK_SIZE, sharedMemSize>>>(d_in, N, d_global_sum);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back
    int h_global_sum = 0;
    CHECK_CUDA(cudaMemcpy(&h_global_sum, d_global_sum, sizeof(int), cudaMemcpyDeviceToHost));

    // Verify and print
    printf("Computed sum = %d, Expected sum = %d\n", h_global_sum, N);

    // Cleanup
    free(h_in);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_global_sum));

    return 0;
}
```