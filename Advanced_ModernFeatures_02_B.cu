```cpp
/*
Aim of the program: 
Use `cooperative_groups::thread_block_tile<32>` to perform a warp-level reduction using warp-synchronous intrinsics, which is more efficient than using shared memory for the last 32 values.

Thinking:
1. The goal is to demonstrate how a warp-level reduction can be performed efficiently using the cooperative_groups library, specifically the `thread_block_tile<32>` abstraction. This allows us to treat each warp as a separate logical thread block tile.
2. We'll create a simple kernel that performs a reduction (sum) over an array. 
3. For the bulk of the reduction, we will use shared memory within a thread block to reduce partial sums. After the shared-memory reduction, the last warp will have the final 32 partial sums, and we will use warp-synchronous intrinsics via `thread_block_tile<32>` to finish the reduction without shared memory.
4. The warp-synchronous intrinsics we will use are the `__shfl_down_sync` family of functions (specifically `__shfl_down_sync(0xFFFFFFFF, value, offset)`), which allow threads within a warp to exchange values efficiently.
5. The kernel will:
   - Load data into shared memory,
   - Perform a tree-based reduction using shared memory,
   - Then, if the thread index is in the last warp (e.g., tid < 32), use `thread_block_tile<32>` to perform the final reduction among the 32 warp threads.
6. The host code will allocate an input array, initialize it, copy it to the device, launch the kernel, and retrieve the final result.
7. To keep the code self-contained, we will perform the reduction on a single block of threads (e.g., 256 threads). For larger arrays, we would launch multiple blocks and combine results on the host or with a second kernel; that complexity is omitted for clarity.
8. The output of the program will print the sum of all elements of the input array, verifying that the warp-level reduction worked correctly.
*/

#include <iostream>
#include <vector>
#include <numeric>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Kernel that performs a sum reduction over an array using shared memory
// and finishes the last warp reduction using thread_block_tile<32>
__global__ void warp_reduce_sum(const float *input, float *output, size_t N)
{
    // Allocate shared memory for block reduction
    extern __shared__ float sdata[];

    // Each thread loads one element (if available) into shared memory
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = 0.0f;
    if (idx < N)
        x = input[idx];

    sdata[tid] = x;
    __syncthreads();

    // Perform block-wide reduction in shared memory
    // Tree reduction: stride halving
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // At this point, the first 32 threads hold partial sums that need final reduction
    // Only the first 32 threads will participate in the warp reduction
    if (tid < 32)
    {
        // Use warp-synchronous reduction with thread_block_tile<32>
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());

        // Perform warp-level reduction using __shfl_down_sync
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other = warp.shfl_down_sync(0xFFFFFFFF, sdata[tid], offset);
            sdata[tid] += other;
        }
    }

    // The first thread writes the block result to global memory
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

int main()
{
    const size_t N = 1 << 20;  // 1M elements
    std::vector<float> h_in(N, 1.0f);  // Initialize all elements to 1.0f
    float h_expected = static_cast<float>(N);  // Expected sum

    // Allocate device memory
    float *d_in, *d_out;
    size_t blockSize = 256;
    size_t gridSize = (N + blockSize - 1) / blockSize;
    cudaMalloc((void**)&d_in, N * sizeof(float));
    cudaMalloc((void**)&d_out, gridSize * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    size_t sharedMemSize = blockSize * sizeof(float);
    warp_reduce_sum<<<gridSize, blockSize, sharedMemSize>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    // Copy partial sums back to host
    std::vector<float> h_out(gridSize);
    cudaMemcpy(h_out.data(), d_out, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Final reduction on host
    float h_result = std::accumulate(h_out.begin(), h_out.end(), 0.0f);

    // Verify
    std::cout << "GPU sum = " << h_result << std::endl;
    std::cout << "CPU expected sum = " << h_expected << std::endl;
    std::cout << "Result correct? " << ((fabs(h_result - h_expected) < 1e-5) ? "Yes" : "No") << std::endl;

    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
```