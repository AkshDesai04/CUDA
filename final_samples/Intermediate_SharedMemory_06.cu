```cpp
/*
Implement a 1D convolution (or a 1D stencil) using shared memory. Each thread block should load a segment of the input array into shared memory, including 'ghost' cells at the boundaries needed by the threads at the edges of the block. Use __syncthreads() before computation.

Thinking:
- The goal is to demonstrate a simple 1D convolution using CUDA shared memory.
- We'll use a fixed kernel width of 5 (radius = 2). The kernel coefficients can be any values; here we use a uniform kernel (each weight = 0.2).
- Each thread block will load its own segment plus the necessary halo (ghost) cells into shared memory. This allows all threads in the block to perform convolution without accessing global memory for each element.
- The shared memory size per block is blockDim.x + 2 * radius. Each thread will load its main element into the middle of this shared array and, if needed, will also load halo elements from neighboring positions.
- After loading, we synchronize with __syncthreads() to ensure all data is available in shared memory before any thread begins the convolution computation.
- Boundary handling: For elements near the global array boundaries, out-of-range accesses are treated as zero (zero padding). This simplifies the code and is common for stencil operations.
- The kernel is launched with a grid that covers the entire input array. Threads that map to out-of-range global indices simply skip writing to the output.
- Error checking macro is included for robustness.
- The host code allocates host input, initializes it, copies it to device, sets the kernel coefficients in device constant memory, launches the kernel, copies back the result, and prints the first few outputs for verification.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                         \
    {                                                                            \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    }

__constant__ float d_kernel[5];  // Kernel coefficients (radius = 2)

// 1D convolution kernel using shared memory
__global__ void conv1d(const float* __restrict__ input,
                       float* __restrict__ output,
                       int N,
                       const float* __restrict__ kernel,
                       int radius)
{
    extern __shared__ float s_data[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load main element into shared memory
    if (gid < N)
        s_data[tid + radius] = input[gid];
    else
        s_data[tid + radius] = 0.0f;

    // Load left halo if needed
    if (tid < radius) {
        int gid_l = gid - radius;
        s_data[tid] = (gid_l >= 0) ? input[gid_l] : 0.0f;
    }

    // Load right halo if needed
    if (tid >= blockDim.x - radius) {
        int gid_r = gid + radius;
        s_data[tid + blockDim.x + radius] = (gid_r < N) ? input[gid_r] : 0.0f;
    }

    __syncthreads();

    // Perform convolution if within bounds
    if (gid < N) {
        float sum = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            sum += kernel[k + radius] * s_data[tid + k + radius];
        }
        output[gid] = sum;
    }
}

int main()
{
    const int N = 1 << 20;          // 1M elements
    const int radius = 2;
    const int kernelSize = 2 * radius + 1;
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Host memory allocation
    float* h_input  = (float*)malloc(N * sizeof(float));
    float* h_output = (float*)malloc(N * sizeof(float));
    float h_kernel[kernelSize] = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};  // Uniform kernel

    // Initialize input with some values
    for (int i = 0; i < N; ++i)
        h_input[i] = static_cast<float>(i) / N;  // Example: ramp

    // Device memory allocation
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc((void**)&d_input,  N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_output, N * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Copy kernel to constant memory
    CHECK_CUDA(cudaMemcpyToSymbol(d_kernel, h_kernel, kernelSize * sizeof(float)));

    // Launch kernel
    size_t sharedMemSize = (threadsPerBlock + 2 * radius) * sizeof(float);
    conv1d<<<blocks, threadsPerBlock, sharedMemSize>>>(d_input, d_output, N, d_kernel, radius);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few outputs
    printf("First 10 output values:\n");
    for (int i = 0; i < 10; ++i)
        printf("%f ", h_output[i]);
    printf("\n");

    // Cleanup
    free(h_input);
    free(h_output);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
```