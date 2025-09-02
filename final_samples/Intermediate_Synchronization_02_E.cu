```cpp
/*
 * Aim: Compare the performance of your two-pass reduction with Thrust's `thrust::reduce` or `thrust::max_element`.
 *
 * Thinking:
 * The goal of this program is to benchmark a hand‑crafted two‑pass reduction kernel against the
 * highly optimised reduction utilities provided by the Thrust library.  We generate a large
 * floating‑point array on the host, copy it to the GPU, then perform the following steps:
 *
 * 1. **Two‑pass reduction**:
 *    - **First kernel** (`reduce_kernel`) reduces chunks of the input array into per‑block
 *      partial sums using shared memory.  Each block writes its result to a `block_sums` array.
 *    - **Second kernel** (`reduce_block_sums`) reduces the `block_sums` array (which is
 *      usually small) to a single result.  This approach keeps the reduction workload in
 *      two stages so that each kernel can be tuned independently.
 *    - Timing is measured with CUDA events to capture only device execution time.
 *
 * 2. **Thrust reduction**:
 *    - We use `thrust::device_vector<float>` to hold the data and call `thrust::reduce` to
 *      compute the sum.  Timing is measured similarly with CUDA events.
 *
 * 3. **Verification**:
 *    - The two results are compared for equality within a small epsilon.  If they disagree,
 *      the program reports an error.
 *
 * 4. **Output**:
 *    - The program prints the time taken by each method in milliseconds and the speed‑up
 *      factor (two‑pass reduction time divided by Thrust time).  A lower time indicates
 *      better performance.  We expect Thrust to be highly optimised, so the hand‑crafted
 *      reduction will usually be slower unless it is carefully tuned.
 *
 * Design decisions:
 * - Block size is fixed at 256 threads; this is a common choice for many GPUs.
 * - We handle array sizes that are not powers of two by padding the input or by checking
 *   bounds in the kernel.
 * - The program uses `cudaMalloc`/`cudaFree` for device memory and `cudaMemcpy` for data
 *   transfer.
 * - Error checking is performed after every CUDA API call via the `checkCudaError` helper.
 * - For reproducibility we seed the random number generator with a fixed value.
 *
 * Compilation:
 *     nvcc -O3 -std=c++11 -arch=sm_70 two_pass_vs_thrust.cu -o reduction_compare
 *
 * Execution:
 *     ./reduction_compare
 *
 * The output will look something like:
 *     Two‑pass reduction time:  3.42 ms
 *     Thrust reduce time:      1.18 ms
 *     Speed‑up factor:         2.89
 *     Results match within tolerance.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

// Macro to check CUDA errors
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            std::cerr << "CUDA error in file '" << __FILE__        \
                      << "' in line " << __LINE__ << ": "          \
                      << cudaGetErrorString(err) << std::endl;     \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// First pass: reduce a segment of the array into per‑block sums
__global__ void reduce_kernel(const float *input, float *block_sums, size_t N) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x; // Stride of 2 for unrolling

    float sum = 0.0f;

    // Load two elements per thread if within bounds
    if (idx < N) sum += input[idx];
    if (idx + blockDim.x < N) sum += input[idx + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    // In‑block reduction (binary tree)
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp reduction (no sync needed)
    if (tid < 32) {
        volatile float *vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // Write result for this block to global memory
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

// Second pass: reduce block_sums array (small) to single result
__global__ void reduce_block_sums(float *block_sums, size_t N) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x;

    // Load into shared memory
    float sum = 0.0f;
    if (idx < N) sum = block_sums[idx];
    sdata[tid] = sum;
    __syncthreads();

    // In‑block reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // First thread writes the result
    if (tid == 0) {
        block_sums[0] = sdata[0];
    }
}

// Helper to time a CUDA kernel using events
float timeKernel(const cudaEvent_t &start, const cudaEvent_t &stop) {
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

int main() {
    const size_t N = 1 << 26; // ~67 million elements (~256 MB)
    const int THREADS_PER_BLOCK = 256;

    // Allocate host memory and initialize data
    std::vector<float> h_data(N);
    srand(42);
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_input = nullptr;
    float *d_block_sums = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_input, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Determine number of blocks for first pass
    size_t blocks_per_grid = (N + THREADS_PER_BLOCK * 2 - 1) / (THREADS_PER_BLOCK * 2);
    CHECK_CUDA(cudaMalloc(&d_block_sums, blocks_per_grid * sizeof(float)));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ----------------- Two‑pass reduction -----------------
    CHECK_CUDA(cudaEventRecord(start));

    // Launch first pass
    size_t shared_mem_bytes = THREADS_PER_BLOCK * sizeof(float);
    reduce_kernel<<<blocks_per_grid, THREADS_PER_BLOCK, shared_mem_bytes>>>(d_input, d_block_sums, N);
    CHECK_CUDA(cudaGetLastError());

    // Launch second pass (small enough to use 1 block)
    size_t second_blocks = 1;
    reduce_block_sums<<<second_blocks, THREADS_PER_BLOCK, shared_mem_bytes>>>(d_block_sums, blocks_per_grid);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float two_pass_time = timeKernel(start, stop);

    // Copy result back to host
    float gpu_two_pass_sum = 0.0f;
    CHECK_CUDA(cudaMemcpy(&gpu_two_pass_sum, d_block_sums, sizeof(float), cudaMemcpyDeviceToHost));

    // ----------------- Thrust reduce -----------------
    // Prepare device vector
    thrust::device_vector<float> d_vec(h_data.begin(), h_data.end());

    CHECK_CUDA(cudaEventRecord(start));
    float thrust_sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float thrust_time = timeKernel(start, stop);

    // ----------------- Verification -----------------
    const float epsilon = 1e-4f;
    bool match = std::abs(gpu_two_pass_sum - thrust_sum) < epsilon;
    if (!match) {
        std::cerr << "Error: Results do not match!" << std::endl;
        std::cerr << "Two‑pass sum: " << gpu_two_pass_sum << ", Thrust sum: " << thrust_sum << std::endl;
    }

    // ----------------- Output -----------------
    std::cout << "Two-pass reduction time:  " << two_pass_time << " ms" << std::endl;
    std::cout << "Thrust reduce time:      " << thrust_time << " ms" << std::endl;
    if (thrust_time > 0.0f) {
        std::cout << "Speed‑up factor:         " << two_pass_time / thrust_time << std::endl;
    }
    std::cout << (match ? "Results match within tolerance." : "Results do NOT match.") << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_block_sums));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
```