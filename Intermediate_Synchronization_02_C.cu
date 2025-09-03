/*
Aim of the program: Modify the algorithm to find not just the maximum value, but also the index of that value. This requires carrying a struct {value, index} through the reduction.

Thinking:
The original reduction algorithm for finding the maximum value only keeps track of the value. To also obtain the index of the maximum element, we need to propagate a pair of (value, index) through the reduction steps. In a CUDA reduction, each thread can load its element and associate its global index. During the binary reduction inside a block, we compare the values of the pairs and keep the one with the larger value, preserving its associated index. The result of each block is a single MaxVal struct that contains the maximum value and its global index for that block.

After the block‑wise reduction, we will copy the block results back to the host and perform a simple linear scan to find the overall maximum and its index. This approach keeps the GPU work minimal (one kernel launch) and leverages the host for the final reduction of a small number of block results.

Implementation details:
- Define a struct MaxVal { float value; int index; }.
- In the kernel, use shared memory to store the per‑thread MaxVal pairs.
- Handle odd number of elements by setting the second element to a sentinel low value if out of bounds.
- Use a typical reduction pattern (stride halving) while comparing values and propagating indices.
- After kernel launch, copy the per‑block MaxVal results to the host and find the global maximum.
- Include basic CUDA error checking macro.
- Generate test data on the host (random floats), copy to device, launch kernel, copy back results, and print the final maximum value and index.

The final .cu file is self‑contained and can be compiled with nvcc.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

struct MaxVal {
    float value;
    int   index;
};

__global__ void reduce_max_index(const float* __restrict__ d_in,
                                 MaxVal* __restrict__ d_out,
                                 int n)
{
    extern __shared__ MaxVal sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Load two elements per thread for better coalescing
    MaxVal local;
    if (idx < n) {
        local.value = d_in[idx];
        local.index = idx;
    } else {
        local.value = -FLT_MAX; // sentinel for empty slots
        local.index = -1;
    }

    // If there is a second element in the same stride
    unsigned int idx2 = idx + stride;
    if (idx2 < n) {
        float val2 = d_in[idx2];
        if (val2 > local.value) {
            local.value = val2;
            local.index = idx2;
        }
    }

    sdata[tid] = local;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            MaxVal other = sdata[tid + s];
            if (other.value > sdata[tid].value) {
                sdata[tid] = other;
            }
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

int main()
{
    const int N = 1 << 20; // 1M elements
    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Host allocation
    float* h_in = (float*)malloc(N * sizeof(float));
    if (!h_in) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input with random values
    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<float>(rand()) / RAND_MAX * 1000.0f;
    }

    // Device allocation
    float* d_in = nullptr;
    MaxVal* d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, blocks * sizeof(MaxVal)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    size_t sharedMemSize = threadsPerBlock * sizeof(MaxVal);
    reduce_max_index<<<blocks, threadsPerBlock, sharedMemSize>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy block results back to host
    MaxVal* h_blockResults = (MaxVal*)malloc(blocks * sizeof(MaxVal));
    if (!h_blockResults) {
        fprintf(stderr, "Failed to allocate host memory for block results.\n");
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaMemcpy(h_blockResults, d_out, blocks * sizeof(MaxVal), cudaMemcpyDeviceToHost));

    // Final reduction on host
    MaxVal globalMax;
    globalMax.value = -FLT_MAX;
    globalMax.index = -1;
    for (int i = 0; i < blocks; ++i) {
        if (h_blockResults[i].value > globalMax.value) {
            globalMax = h_blockResults[i];
        }
    }

    printf("Maximum value: %f at index %d\n", globalMax.value, globalMax.index);

    // Cleanup
    free(h_in);
    free(h_blockResults);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
