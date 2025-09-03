```cpp
/*
Write a program that uses Cooperative Groups to perform a grid-wide synchronization.
Launch a kernel and get the `grid_group`. Use the group's `sync()` method to ensure all threads in the entire grid have reached a certain point before proceeding.
This allows for single-kernel reductions on large arrays.

Thinking:

To demonstrate grid-wide synchronization we will perform a reduction on a large array using
a single kernel launch. CUDA Cooperative Groups provide a `grid_group` that allows all
threads in the grid to synchronize. We need to use `cudaLaunchCooperativeKernel` to
ensure the device supports cooperative launches. The kernel will:

1. Load input elements into shared memory and perform a block-level reduction.
2. Write each block's partial sum to a global array.
3. Use `grid_group.sync()` to make sure all blocks have finished writing.
4. Let one block (block 0) perform a final reduction over the partial sums.
5. Write the final result back to global memory.

The host code will:

- Allocate device memory for the input and a small array to hold partial sums.
- Initialize the input array on the host and copy it to the device.
- Query the device for cooperative launch capability.
- Launch the kernel cooperatively.
- Copy the result back and print it.

The code below follows this design, using CUDA 11+ cooperative groups API, and
includes basic error checking and device capability querying. The program can be
compiled with `nvcc -arch=sm_70 -o grid_sync_reduce grid_sync_reduce.cu` (or a
higher architecture that supports cooperative launch).
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;

// Kernel that reduces an array using cooperative groups for grid-wide sync
__global__ void reduce_kernel(const float *in, float *out, int N)
{
    // Get the grid group for cooperative sync
    cg::grid_group grid = cg::this_grid();

    // Shared memory for block reduction
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory (handle bounds)
    float val = 0.0f;
    if (idx < N)
        val = in[idx];
    sdata[tid] = val;
    __syncthreads();

    // Block-level reduction (tree-based)
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write block's partial sum to global array
    if (tid == 0)
        out[blockIdx.x] = sdata[0];

    // Wait until all blocks have written their partial sums
    grid.sync();

    // Final reduction over block results (performed by block 0)
    if (blockIdx.x == 0)
    {
        // Load block results into shared memory
        if (tid < gridDim.x)
            sdata[tid] = out[tid];
        else
            sdata[tid] = 0.0f;
        __syncthreads();

        for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
        {
            if (tid < s)
                sdata[tid] += sdata[tid + s];
            __syncthreads();
        }

        if (tid == 0)
            out[0] = sdata[0];
    }
}

// Utility macro for CUDA error checking
#define CUDA_CHECK(err)                                                     \
    do {                                                                    \
        cudaError_t err__ = (err);                                          \
        if (err__ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",     \
                    __FILE__, __LINE__, err__, cudaGetErrorString(err__),  \
                    #err);                                                 \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

int main()
{
    const int N = 1 << 20; // 1M elements
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Allocate host memory
    float *h_in = (float*)malloc(N * sizeof(float));
    float h_sum = 0.0f;
    for (int i = 0; i < N; ++i)
    {
        h_in[i] = 1.0f; // Simple data: all ones
        h_sum += h_in[i];
    }

    // Allocate device memory
    float *d_in = nullptr;
    float *d_partial = nullptr; // Holds block partial sums
    CUDA_CHECK(cudaMalloc((void**)&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_partial, gridSize * sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // Check if device supports cooperative launch
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    int cooperativeLaunch;
    CUDA_CHECK(cudaDeviceGetAttribute(&cooperativeLaunch,
        cudaDevAttrCooperativeLaunch, device));

    if (!cooperativeLaunch)
    {
        fprintf(stderr, "Device does not support cooperative launch.\n");
        exit(EXIT_FAILURE);
    }

    // Launch kernel cooperatively
    void *kernelArgs[] = { &d_in, &d_partial, &N };
    size_t sharedMemBytes = blockSize * sizeof(float);

    // Note: For cooperative launch, number of blocks must be <= maximum grid size for cooperative groups
    int maxBlocks = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&maxBlocks,
        cudaDevAttrMaxCooperativeGroupCount, device));
    if (gridSize > maxBlocks)
    {
        fprintf(stderr, "Requested grid size %d exceeds device limit %d for cooperative groups.\n",
                gridSize, maxBlocks);
        exit(EXIT_FAILURE);
    }

    cudaLaunchCooperativeKernel((void*)reduce_kernel, gridSize, blockSize,
                                kernelArgs, sharedMemBytes, 0);
    CUDA_CHECK(cudaGetLastError());

    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    float h_result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_result, d_partial, sizeof(float), cudaMemcpyDeviceToHost));

    printf("Expected sum: %f, GPU sum: %f\n", h_sum, h_result);

    // Clean up
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_partial));
    free(h_in);

    return 0;
}
```