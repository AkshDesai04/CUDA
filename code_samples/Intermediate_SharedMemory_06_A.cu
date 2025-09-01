```cpp
/*********************************************************************
 * Aim: For a 3-point stencil `output[i] = w_l*in[i-1] + w_c*in[i] + w_r*in[i+1]`,
 * each thread needs its neighbors. If a block loads N elements for N threads,
 * it must actually load N+2 elements into shared memory.
 *
 * Thinking:
 *   - To demonstrate this, we write a CUDA kernel that applies a
 *     3‑point stencil to an array. Each block will load its
 *     segment of the input array plus one halo element on each side
 *     into shared memory, giving N+2 elements for N threads.
 *   - Each thread then reads the three consecutive elements from
 *     shared memory to compute the stencil for its global index.
 *   - The host code allocates a small example array, initializes it,
 *     copies it to the device, launches the kernel, copies back the
 *     result, and prints a few values to verify correctness.
 *   - Simple timing using CUDA events is included to illustrate
 *     performance, though the array size is small for brevity.
 *
 * The code is self‑contained and can be compiled with:
 *     nvcc -arch=sm_70 -o stencil stencil.cu
 *********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Simple macro for CUDA error checking
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",        \
                    __func__, __FILE__, __LINE__,                  \
                    cudaGetErrorString(err));                       \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Device kernel performing 3‑point stencil with shared memory
__global__ void stencil_kernel(const float *in, float *out,
                               const float w_l, const float w_c,
                               const float w_r, const int N)
{
    // Shared memory: blockDim.x elements + 2 halo elements
    extern __shared__ float s[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load main data into shared memory at index tid+1
    if (gid < N)
        s[tid + 1] = in[gid];
    else
        s[tid + 1] = 0.0f;  // padding for out-of-bounds

    // Load left halo for the first thread in the block
    if (tid == 0) {
        int left_gid = gid - 1;
        s[0] = (left_gid >= 0) ? in[left_gid] : 0.0f;
    }

    // Load right halo for the last thread in the block
    if (tid == blockDim.x - 1) {
        int right_gid = gid + 1;
        s[blockDim.x + 1] = (right_gid < N) ? in[right_gid] : 0.0f;
    }

    __syncthreads();

    // Perform stencil for valid indices (skip boundaries)
    if (gid > 0 && gid < N - 1) {
        out[gid] = w_l * s[tid] + w_c * s[tid + 1] + w_r * s[tid + 2];
    }
}

// Host function to verify correctness
void cpu_stencil(const float *in, float *out, const float w_l,
                 const float w_c, const float w_r, const int N)
{
    for (int i = 1; i < N - 1; ++i) {
        out[i] = w_l * in[i - 1] + w_c * in[i] + w_r * in[i + 1];
    }
}

int main(void)
{
    const int N = 1024;                 // Size of the array
    const int threadsPerBlock = 256;    // Number of threads per block
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host memory
    float *h_in  = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)malloc(N * sizeof(float));
    float *h_cpu = (float*)malloc(N * sizeof(float));

    // Initialize input with some values (e.g., sin wave)
    for (int i = 0; i < N; ++i) {
        h_in[i] = sinf(2.0f * 3.14159265f * i / N);
    }

    // Allocate device memory
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, N * sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // Weights for the stencil
    const float w_l = 0.25f;
    const float w_c = 0.5f;
    const float w_r = 0.25f;

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Launch kernel
    CUDA_CHECK(cudaEventRecord(start));
    size_t sharedMemSize = (threadsPerBlock + 2) * sizeof(float);
    stencil_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_in, d_out,
                                                                     w_l, w_c, w_r, N);
    CUDA_CHECK(cudaEventRecord(stop));

    // Wait for completion
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute CPU reference
    cpu_stencil(h_in, h_cpu, w_l, w_c, w_r, N);

    // Verify results (allow small tolerance)
    int errors = 0;
    for (int i = 1; i < N - 1; ++i) {
        float diff = fabsf(h_out[i] - h_cpu[i]);
        if (diff > 1e-5f) {
            errors++;
            if (errors <= 10) {
                printf("Mismatch at i=%d: GPU=%f CPU=%f diff=%f\n",
                       i, h_out[i], h_cpu[i], diff);
            }
        }
    }
    if (errors == 0) {
        printf("All results match within tolerance.\n");
    } else {
        printf("Total mismatches: %d\n", errors);
    }

    // Print timing
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %.3f ms\n", milliseconds);

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);
    free(h_cpu);

    return 0;
}
```