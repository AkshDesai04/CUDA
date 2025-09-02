```cpp
/*********************************************************************
 * Aim of the program:
 * Use `cooperative_groups::thread_block_tile<32>` to perform a warp-level reduction using warp-synchronous intrinsics, which is more efficient than using shared memory for the last 32 values.
 *
 * Thinking:
 * 1. The goal is to demonstrate a hybrid reduction strategy that uses shared memory for most of the work
 *    but switches to warp-synchronous intrinsics (shuffles) when only a warp (32 threads) is left.
 * 2. `cooperative_groups` provides the `thread_block_tile<32>` type that represents a warp inside
 *    a block.  Its `reduce` method uses shuffle instructions internally, which are faster than
 *    writing back to shared memory and synchronizing.
 * 3. In the kernel we:
 *    - Load one element per thread into shared memory (`sdata`).
 *    - Perform pairwise reductions in shared memory, halving the active threads each time, until
 *      only 32 threads remain.  This is done with a simple for-loop and `__syncthreads()`.
 *    - When 32 threads remain we create a warp object via `cg::tiled_partition<32>`.
 *    - Each of the 32 threads loads its value from shared memory into a local variable.
 *    - We call `warp.reduce` with `std::plus<int>()` to sum the 32 values using shuffles.
 *    - Thread 0 writes the block‑level sum to global memory.
 * 4. On the host we launch the kernel, gather block sums, and compute the final global sum.
 * 5. The code includes basic error checking and prints the result.
 *********************************************************************/

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>
#include <functional>
#include <random>

namespace cg = cooperative_groups;

// Error checking macro
#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error in file '" << __FILE__                      \
                      << "' in line " << __LINE__ << ": "                        \
                      << cudaGetErrorString(err) << std::endl;                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

// Kernel: hybrid reduction using shared memory + warp-level reduction
__global__ void reduce_kernel(const int *input, int *block_sums, int N)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    // Load input element or zero if out of bounds
    int val = (idx < N) ? input[idx] : 0;
    sdata[tid] = val;
    __syncthreads();

    // Reduce in shared memory until 32 threads remain
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Only 32 threads are active here. Use warp-level reduction.
    if (tid < 32)
    {
        int warpSum = sdata[tid];
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
        warpSum = warp.reduce(warpSum, std::plus<int>());
        if (tid == 0)
        {
            block_sums[blockIdx.x] = warpSum;
        }
    }
}

int main()
{
    const int N = 1 << 20;          // 1M elements
    const int threadsPerBlock = 128;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host memory
    std::vector<int> h_input(N);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(1, 10);
    for (int &x : h_input)
        x = dist(rng);

    // Allocate device memory
    int *d_input = nullptr;
    int *d_block_sums = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_block_sums, blocks * sizeof(int)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    size_t sharedMemBytes = threadsPerBlock * sizeof(int);
    reduce_kernel<<<blocks, threadsPerBlock, sharedMemBytes>>>(d_input, d_block_sums, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy block sums back to host
    std::vector<int> h_block_sums(blocks);
    CHECK_CUDA(cudaMemcpy(h_block_sums.data(), d_block_sums, blocks * sizeof(int), cudaMemcpyDeviceToHost));

    // Final reduction on host
    long long total = 0;
    for (int x : h_block_sums)
        total += x;

    std::cout << "GPU computed sum: " << total << std::endl;

    // Verify with CPU
    long long cpu_total = 0;
    for (int x : h_input)
        cpu_total += x;
    std::cout << "CPU computed sum: " << cpu_total << std::endl;

    // Clean up
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_block_sums));

    return 0;
}
```