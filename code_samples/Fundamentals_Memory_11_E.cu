```cuda
/*
Aim of the program:
Demonstrate the performance benefit: write two kernels, one where each thread reads a configuration parameter from global memory,
and one where it reads from constant memory. Time both.

Thinking and design decisions:
- Use a simple element-wise multiplication: output[i] = data[i] * config[i].
- `config` is an array of floats. For the global kernel, `config` resides in global memory.
- For the constant kernel, we copy the same array into constant memory using `cudaMemcpyToSymbol`.
- Constant memory has a dedicated cache (size 64 KB). We choose the array size to fit within that limit (16 384 floats = 64 KB).
- To make the performance difference noticeable, we launch enough threads (e.g., 1 048 576 elements) and use many blocks.
- Timing is performed with CUDA events for accurate device-side measurement.
- After execution, we compute and print the elapsed times and the speedup factor.
- The code includes error checking for CUDA API calls and kernel launches.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Size of the configuration array: 16384 floats = 64 KB, fits in constant memory.
#define CONFIG_SIZE 16384

// Total number of elements for the computation
#define TOTAL_ELEMS (1 << 20)  // 1,048,576 elements

// Constant memory declaration
__constant__ float d_configConst[CONFIG_SIZE];

// Global memory kernel: reads config from global memory
__global__ void kernelGlobal(float* out, const float* in, const float* config, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = in[idx] * config[idx % CONFIG_SIZE];
    }
}

// Constant memory kernel: reads config from constant memory
__global__ void kernelConstant(float* out, const float* in, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = in[idx] * d_configConst[idx % CONFIG_SIZE];
    }
}

// Utility macro for checking CUDA errors
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

int main()
{
    // Allocate host memory
    float *h_in = (float*)malloc(TOTAL_ELEMS * sizeof(float));
    float *h_out = (float*)malloc(TOTAL_ELEMS * sizeof(float));
    float *h_config = (float*)malloc(CONFIG_SIZE * sizeof(float));

    // Initialize input data and config
    for (int i = 0; i < TOTAL_ELEMS; ++i) h_in[i] = 1.0f + (float)i * 0.001f;
    for (int i = 0; i < CONFIG_SIZE; ++i) h_config[i] = 0.5f + (float)i * 0.0001f;

    // Device pointers
    float *d_in = nullptr, *d_out = nullptr, *d_config = nullptr;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_in,     TOTAL_ELEMS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out,    TOTAL_ELEMS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_config, CONFIG_SIZE * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_in,     h_in,  TOTAL_ELEMS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_config, h_config, CONFIG_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Copy config to constant memory
    CHECK_CUDA(cudaMemcpyToSymbol(d_configConst, h_config, CONFIG_SIZE * sizeof(float)));

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid    = (TOTAL_ELEMS + threadsPerBlock - 1) / threadsPerBlock;

    // Events for timing
    cudaEvent_t startGlobal, stopGlobal, startConst, stopConst;
    CHECK_CUDA(cudaEventCreate(&startGlobal));
    CHECK_CUDA(cudaEventCreate(&stopGlobal));
    CHECK_CUDA(cudaEventCreate(&startConst));
    CHECK_CUDA(cudaEventCreate(&stopConst));

    // ----------------------------
    // Launch global memory kernel
    // ----------------------------
    CHECK_CUDA(cudaEventRecord(startGlobal));
    kernelGlobal<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, d_config, TOTAL_ELEMS);
    CHECK_CUDA(cudaEventRecord(stopGlobal));
    CHECK_CUDA(cudaEventSynchronize(stopGlobal));

    // ----------------------------
    // Launch constant memory kernel
    // ----------------------------
    CHECK_CUDA(cudaEventRecord(startConst));
    kernelConstant<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, TOTAL_ELEMS);
    CHECK_CUDA(cudaEventRecord(stopConst));
    CHECK_CUDA(cudaEventSynchronize(stopConst));

    // Check for kernel errors
    CHECK_CUDA(cudaGetLastError());

    // Compute elapsed times
    float msGlobal, msConst;
    CHECK_CUDA(cudaEventElapsedTime(&msGlobal, startGlobal, stopGlobal));
    CHECK_CUDA(cudaEventElapsedTime(&msConst, startConst, stopConst));

    // Print results
    printf("Global memory kernel time:   %6.3f ms\n", msGlobal);
    printf("Constant memory kernel time: %6.3f ms\n", msConst);
    if (msGlobal > 0.0f)
        printf("Speedup (global/constant):   %.2fx\n", msGlobal / msConst);
    else
        printf("Speedup (global/constant):   N/A\n");

    // Cleanup
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_config));
    free(h_in);
    free(h_out);
    free(h_config);

    CHECK_CUDA(cudaEventDestroy(startGlobal));
    CHECK_CUDA(cudaEventDestroy(stopGlobal));
    CHECK_CUDA(cudaEventDestroy(startConst));
    CHECK_CUDA(cudaEventDestroy(stopConst));

    return 0;
}
```