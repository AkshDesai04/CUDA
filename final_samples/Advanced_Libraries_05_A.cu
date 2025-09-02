```cpp
/*
 * Aim: Use CUB's `DeviceReduce::Sum` to perform a sum reduction and compare its performance to your own and Thrust's.
 *
 * Thinking:
 * 1. We want a complete, self‑contained CUDA program that:
 *    - Generates a large array of floating point values on the host.
 *    - Copies it to device memory.
 *    - Runs three different summation approaches:
 *      a) A hand‑written parallel reduction kernel.
 *      b) CUB's DeviceReduce::Sum (single‑call API).
 *      c) Thrust's `thrust::reduce` on a device_vector.
 *    - Times each method using CUDA events.
 *    - Prints the results and the timings.
 *
 * 2. Device kernel design:
 *    - Use a standard reduction pattern with shared memory.
 *    - Each block will sum its chunk and write a partial sum to a global array.
 *    - After launching the kernel, perform a second kernel launch to sum the partial sums (or let the same kernel handle multiple stages if data fits).
 *    - For simplicity, we use a two‑stage reduction: first launch `reductionKernel` to produce block sums, then a second launch of `reductionKernel` (or a tiny kernel) to finish the reduction.
 *
 * 3. CUB usage:
 *    - Query temporary storage size via `cub::DeviceReduce::Sum::GetTempStorageSize`.
 *    - Allocate temporary storage.
 *    - Call `cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, d_in, d_out, n)`.
 *
 * 4. Thrust usage:
 *    - Wrap device pointer in `thrust::device_ptr` or use `thrust::device_vector`.
 *    - Call `thrust::reduce(thrust::device, d_in, d_in + n, 0.0f)`.
 *
 * 5. Timing:
 *    - Use `cudaEventCreate`, `cudaEventRecord`, `cudaEventElapsedTime`.
 *    - Wrap each method in a function that returns elapsed time and result.
 *
 * 6. Error checking:
 *    - Provide a macro `CHECK_CUDA` to catch runtime errors.
 *
 * 7. Output:
 *    - Print the computed sum (should be identical for all three methods).
 *    - Print the time taken in milliseconds.
 *
 * 8. Build:
 *    - The program can be compiled with: `nvcc -o sum_compare sum_compare.cu -lcudart -lthrust`.
 *
 * 9. Limitations:
 *    - This example uses float. For larger arrays, the two‑stage reduction might need to be adapted.
 *    - The custom kernel uses a fixed block size; for very large arrays you might want to launch multiple stages.
 *
 * The code below implements all of the above.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__     \
                      << " : " << cudaGetErrorString(err) << std::endl;     \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Custom parallel reduction kernel
__global__ void reductionKernel(const float* __restrict__ d_in, float* d_out, size_t n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Each thread loads two elements (if available) into shared memory
    float sum = 0.0f;
    if (i < n) sum += d_in[i];
    if (i + blockDim.x < n) sum += d_in[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block's result to global memory
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

// Helper to perform full reduction using the custom kernel
float customReduction(const float* d_in, size_t n, cudaStream_t stream = 0) {
    int blockSize = 256; // threads per block
    int gridSize = (n + blockSize * 2 - 1) / (blockSize * 2);

    // Allocate intermediate buffer
    float* d_partial;
    CHECK_CUDA(cudaMalloc(&d_partial, gridSize * sizeof(float)));

    size_t smemSize = blockSize * sizeof(float);
    // First stage
    reductionKernel<<<gridSize, blockSize, smemSize, stream>>>(d_in, d_partial, n);
    CHECK_CUDA(cudaGetLastError());

    // If necessary, perform additional stages
    size_t currentSize = gridSize;
    while (currentSize > 1) {
        int nextGrid = (currentSize + blockSize * 2 - 1) / (blockSize * 2);
        reductionKernel<<<nextGrid, blockSize, smemSize, stream>>>(d_partial, d_partial, currentSize);
        CHECK_CUDA(cudaGetLastError());
        currentSize = nextGrid;
    }

    float result;
    CHECK_CUDA(cudaMemcpyAsync(&result, d_partial, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaFree(d_partial));
    return result;
}

// Timing wrapper
template <typename Func>
float measureTime(Func f, std::string name) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, 0));
    f();
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    std::cout << name << " took " << ms << " ms" << std::endl;
    return ms;
}

int main() {
    const size_t N = 1 << 26; // about 67 million elements (~256 MB)
    std::vector<float> h_data(N);
    // Initialize host data
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i % 100);
    }

    // Device memory
    float* d_in;
    CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // ---- Custom reduction ----
    float customResult = 0.0f;
    measureTime([&]() { customResult = customReduction(d_in, N); }, "Custom Reduction");

    // ---- CUB reduction ----
    float cubResult = 0.0f;
    size_t temp_storage_bytes = 0;
    // Determine temporary device storage requirements
    cub::DeviceReduce::Sum::GetTempStorageSize(&temp_storage_bytes, d_in, &cubResult, N);
    void* d_temp_storage = nullptr;
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    measureTime([&]() {
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, &cubResult, N);
        CHECK_CUDA(cudaGetLastError());
    }, "CUB DeviceReduce::Sum");

    CHECK_CUDA(cudaFree(d_temp_storage));

    // ---- Thrust reduction ----
    float thrustResult = 0.0f;
    measureTime([&]() {
        thrust::device_ptr<float> dev_ptr(d_in);
        thrustResult = thrust::reduce(dev_ptr, dev_ptr + N, 0.0f, thrust::plus<float>());
    }, "Thrust reduce");

    // Verify results
    std::cout << "Custom Result:  " << customResult  << std::endl;
    std::cout << "CUB Result:     " << cubResult     << std::endl;
    std::cout << "Thrust Result:  " << thrustResult  << std::endl;

    // Clean up
    CHECK_CUDA(cudaFree(d_in));
    return 0;
}
```