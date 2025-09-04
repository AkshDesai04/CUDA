```cpp
/*
Aim of the program:
Use CUB's `DeviceReduce::Sum` to perform a sum reduction and compare its performance to your own and Thrust's.

Thinking:
To compare three reduction implementations – a hand‑written CUDA kernel, the Thrust library, and CUB's DeviceReduce – I plan to:

1. Generate a large array of random floats on the host and copy it to the device.
2. Perform a reduction with:
   a) A custom CUDA kernel that reduces the array in multiple stages (blockwise partial sums, then further reduction until one value remains).
   b) Thrust's `thrust::reduce` on a `thrust::device_vector`.
   c) CUB's `cub::DeviceReduce::Sum` in a single launch, allocating temporary storage as required.
3. Time each approach using CUDA events to get accurate GPU timings.
4. Verify correctness by comparing the results of all three methods.
5. Print the size, time for each method, the resulting sum, and the relative speedups.

The custom kernel uses a shared‑memory reduction pattern. To keep it simple, it repeatedly launches a block‑wise reduction kernel until the array is reduced to a single element, swapping input/output pointers each iteration. Thrust and CUB are straightforward wrappers around efficient reduction kernels.

The program will compile with `nvcc -lcub -lthrust`.*/

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

// Macro for CUDA error checking
#define CHECK_CUDA(call)                                           \
    {                                                              \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",      \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    }

// Block size for reduction kernels
const int BLOCK_SIZE = 256;

// Custom reduction kernel: each block reduces a chunk of the input array into a single value in the output array.
__global__ void blockReduceKernel(const float *d_in, float *d_out, size_t n)
{
    __shared__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;

    // Each thread loads up to two elements
    if (idx < n) sum += d_in[idx];
    if (idx + blockDim.x < n) sum += d_in[idx + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    // Intra-block reduction
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Warp-level reduction (32 threads or less)
    if (tid < 32) {
        volatile float *vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // Write result of this block to output
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

// Host wrapper for custom reduction
float customReduction(const float *d_in, size_t n)
{
    size_t numElements = n;
    const float *d_input = d_in;
    float *d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_output, ((numElements + BLOCK_SIZE - 1) / BLOCK_SIZE) * sizeof(float)));

    const int grid = (numElements + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    const int block = BLOCK_SIZE;

    // Reduction loop
    while (numElements > 1) {
        blockReduceKernel<<<grid, block>>>(d_input, d_output, numElements);
        CHECK_CUDA(cudaGetLastError());
        // Prepare for next iteration
        numElements = grid;
        // Swap input and output pointers
        const float *tmp = d_input;
        d_input = d_output;
        d_output = (float*)tmp;
        // Recompute grid size for next iteration
        grid = (numElements + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    }

    // Result is in d_input[0]
    float h_result = 0.0f;
    CHECK_CUDA(cudaMemcpy(&h_result, d_input, sizeof(float), cudaMemcpyDeviceToHost));
    // Free temporary buffer (the one not used in final iteration)
    CHECK_CUDA(cudaFree(d_output));
    return h_result;
}

// Helper to time a kernel or function using CUDA events
float timeKernel(void (*func)(void*), void *arg)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    func(arg);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms;
}

// Wrapper to pass custom reduction to timing helper
struct CustomReductArg {
    const float *d_in;
    size_t n;
};

void customReductWrapper(void *arg)
{
    CustomReductArg *a = static_cast<CustomReductArg*>(arg);
    customReduction(a->d_in, a->n);
}

// Thrust reduction wrapper
struct ThrustReductArg {
    const float *d_in;
    size_t n;
};

void thrustReductWrapper(void *arg)
{
    ThrustReductArg *a = static_cast<ThrustReductArg*>(arg);
    thrust::device_ptr<const float> dptr(a->d_in);
    thrust::reduce(dptr, dptr + a->n, 0.0f, thrust::plus<float>());
}

int main()
{
    const size_t N = 1 << 24; // 16M elements (~64MB for float)
    printf("Reducing %zu elements.\n", N);

    // Allocate host memory and generate random data
    float *h_data = (float*)malloc(N * sizeof(float));
    srand((unsigned)time(nullptr));
    for (size_t i = 0; i < N; ++i)
        h_data[i] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate device memory
    float *d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

    // 1. Custom reduction
    float custom_time = timeKernel(customReductWrapper, (void*)new CustomReductArg{d_data, N});
    float custom_result;
    // Retrieve result from device (in d_data[0] after last iteration)
    CHECK_CUDA(cudaMemcpy(&custom_result, d_data, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Custom reduction: %f ms, result = %f\n", custom_time, custom_result);

    // 2. Thrust reduction
    float thrust_time = timeKernel(thrustReductWrapper, (void*)new ThrustReductArg{d_data, N});
    float thrust_result;
    CHECK_CUDA(cudaMemcpy(&thrust_result, d_data, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Thrust reduction: %f ms, result = %f\n", thrust_time, thrust_result);

    // 3. CUB reduction
    float cub_time;
    float cub_result;
    // Allocate output
    float *d_cub_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_cub_out, sizeof(float)));

    // Determine temporary storage size
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_data, d_cub_out, N);
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Timing CUB
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_data, d_cub_out, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&cub_time, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaMemcpy(&cub_result, d_cub_out, sizeof(float), cudaMemcpyDeviceToHost));
    printf("CUB reduction: %f ms, result = %f\n", cub_time, cub_result);

    // Verify results are close
    const float eps = 1e-5f;
    if (fabs(custom_result - thrust_result) > eps) {
        printf("Error: Custom and Thrust results differ!\n");
    }
    if (fabs(custom_result - cub_result) > eps) {
        printf("Error: Custom and CUB results differ!\n");
    }

    // Cleanup
    delete (CustomReductArg*)customReductWrapper;
    delete (ThrustReductArg*)thrustReductWrapper;
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_cub_out));
    CHECK_CUDA(cudaFree(d_temp_storage));
    free(h_data);

    return 0;
}
```