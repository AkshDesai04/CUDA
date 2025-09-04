/*
Aim: To use cooperative groups for grid-wide sync, the kernel must be launched with `cudaLaunchCooperativeKernel`. Adapt your launch code.

Thinking:
The task requires a CUDA program that demonstrates the use of cooperative groups for a grid‑wide synchronization. The kernel must be launched with `cudaLaunchCooperativeKernel` instead of the usual `<<<>>>` syntax.  
The simplest example that requires a grid‑wide sync is a parallel reduction (sum) over a large array. Each block can reduce its chunk into a per‑block partial sum, but to obtain the final sum we need to synchronize all blocks so that the per‑block results are ready before a single block aggregates them.  
CUDA cooperative groups provide a `grid_group` that can perform `grid.sync()` across all threads in all blocks. To enable a kernel to be launched cooperatively we must set the `cudaFuncAttributeCooperativeLaunch` attribute on the kernel function and use `cudaLaunchCooperativeKernel`.  
The program below:
1. Allocates a large input array on the host and copies it to the device.  
2. Allocates device memory for per‑block partial sums and a single integer to hold the final result.  
3. Sets the cooperative launch attribute on the kernel.  
4. Launches the kernel using `cudaLaunchCooperativeKernel`.  
5. Waits for completion, copies the final sum back to the host, and prints it.  
The kernel performs a per‑block reduction, stores each block’s partial sum into global memory, synchronizes the grid, and then the first block aggregates all partial sums into the final result. This demonstrates a real use of grid‑wide sync with cooperative groups.

*/

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

// Kernel that performs a parallel sum using grid‑wide synchronization
__global__ void cooperativeSumKernel(const int* input, int* blockSums, int* result, int N)
{
    // 1D thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread sums its strided portion of the array
    int localSum = 0;
    for (int i = idx; i < N; i += stride)
        localSum += input[i];

    // Reduction within block using shared memory
    extern __shared__ int sdata[];
    sdata[threadIdx.x] = localSum;
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    // Write block result to global memory
    if (threadIdx.x == 0)
        blockSums[blockIdx.x] = sdata[0];

    // Grid‑wide synchronization to ensure all blockSums are ready
    grid_group grid = this_grid();
    grid.sync();

    // First block aggregates block sums into final result
    if (blockIdx.x == 0)
    {
        int total = 0;
        for (int i = 0; i < gridDim.x; ++i)
            total += blockSums[i];
        *result = total;
    }
}

// Helper macro for error checking
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main()
{
    // Size of the input array
    const int N = 1 << 20;          // 1M elements
    const size_t bytes = N * sizeof(int);

    // Allocate host memory
    int *h_in = (int*)malloc(bytes);
    if (!h_in) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host array with random integers
    for (int i = 0; i < N; ++i)
        h_in[i] = 1;  // For simplicity, use 1 so the sum is known

    // Allocate device memory
    int *d_in, *d_blockSums, *d_result;
    CUDA_CHECK(cudaMalloc((void**)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_blockSums, sizeof(int) * 4)); // 4 blocks
    CUDA_CHECK(cudaMalloc((void**)&d_result, sizeof(int)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Set cooperative launch attribute on the kernel
    CUDA_CHECK(cudaFuncSetAttribute((void*)cooperativeSumKernel,
                                    cudaFuncAttributeCooperativeLaunch,
                                    1));

    // Launch parameters
    dim3 blockDim(256);          // threads per block
    dim3 gridDim(4);             // number of blocks

    // Compute shared memory size (blockDim.x ints)
    size_t sharedMemBytes = blockDim.x * sizeof(int);

    // Launch cooperative kernel
    void *kernelArgs[] = {(void*)&d_in, (void*)&d_blockSums, (void*)&d_result, (void*)&N};
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)cooperativeSumKernel,
                                           gridDim, blockDim,
                                           kernelArgs,
                                           sharedMemBytes,
                                           0));

    // Wait for completion
    CUDA_CHECK(cudaDeviceSynchronize());

    // Retrieve result
    int h_result = 0;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    printf("Computed sum = %d, Expected sum = %d\n", h_result, N);

    // Clean up
    cudaFree(d_in);
    cudaFree(d_blockSums);
    cudaFree(d_result);
    free(h_in);

    return 0;
}
