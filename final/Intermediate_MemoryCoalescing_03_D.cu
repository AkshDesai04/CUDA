```cpp
/* 
Aim of the program (word to word): 
Plot the effective bandwidth (GB/s) of both kernels for various matrix sizes. 

Thinking process:
1. The goal is to benchmark two simple CUDA kernels that perform a memory copy operation
   (src -> dst) and compute the effective memory bandwidth for different square matrix sizes.
2. For each matrix size, we allocate device arrays `src` and `dst` of type float, each of
   dimension N × N. We initialize `src` with some data (e.g., using a simple pattern) and
   leave `dst` uninitialized.
3. Two kernels are defined:
   - `kernel_naive`: each thread reads one element from `src` and writes it to `dst`.
     This represents a baseline copy operation with global memory access only.
   - `kernel_shared`: each thread loads a tile of elements into shared memory and then
     writes them back to `dst`. This introduces shared memory usage and potential
     benefits for certain memory access patterns. While shared memory does not
     necessarily increase bandwidth for a pure copy, it demonstrates how different
     memory hierarchies can affect performance.
4. We measure execution time of each kernel using CUDA events (`cudaEventRecord` /
   `cudaEventSynchronize`) and compute the time in milliseconds.
5. Effective bandwidth is computed as:
      bandwidth (GB/s) = (2 × N × N × sizeof(float)) / (time_in_seconds) / 1e9
   The factor 2 accounts for one read and one write of the data.
6. The program runs a set of matrix sizes (e.g., 256, 512, 1024, 2048, 4096), prints
   the results in a tabular format. The output can be redirected to a file and then
   plotted externally (e.g., with gnuplot or Python matplotlib).
7. The code includes basic error checking for CUDA API calls and uses a
   simple error macro for brevity.
8. The program is self‑contained in a single .cu file and compiles with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel 1: Naive global memory copy
__global__ void kernel_naive(const float* __restrict__ src, float* __restrict__ dst, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (idx < total)
    {
        dst[idx] = src[idx];
    }
}

// Kernel 2: Copy using shared memory (for demonstration)
__global__ void kernel_shared(const float* __restrict__ src, float* __restrict__ dst, int N)
{
    extern __shared__ float tile[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (idx < total)
    {
        tile[threadIdx.x] = src[idx];   // Load into shared memory
        __syncthreads();                // Ensure all loads complete
        dst[idx] = tile[threadIdx.x];   // Write back from shared memory
    }
}

int main()
{
    // Define matrix sizes to test
    const int sizes[] = {256, 512, 1024, 2048, 4096};
    const int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    // Thread/block configuration
    const int threadsPerBlock = 256;

    printf("%10s %15s %15s\n", "MatrixSize", "BandwidthNaive(GB/s)", "BandwidthShared(GB/s)");
    for (int s = 0; s < numSizes; ++s)
    {
        int N = sizes[s];
        size_t bytes = (size_t)N * N * sizeof(float);

        // Allocate host memory
        float* h_src = (float*)malloc(bytes);
        float* h_dst = (float*)malloc(bytes);
        if (!h_src || !h_dst)
        {
            fprintf(stderr, "Host memory allocation failed for size %d\n", N);
            exit(EXIT_FAILURE);
        }

        // Initialize source with some data
        for (int i = 0; i < N * N; ++i)
            h_src[i] = (float)i;

        // Allocate device memory
        float *d_src, *d_dst;
        CUDA_CHECK(cudaMalloc((void**)&d_src, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_dst, bytes));

        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));

        // Define grid size
        int blocks = (N * N + threadsPerBlock - 1) / threadsPerBlock;

        // Events for timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // ----------------- Kernel 1: Naive -----------------
        CUDA_CHECK(cudaEventRecord(start, 0));
        kernel_naive<<<blocks, threadsPerBlock>>>(d_src, d_dst, N);
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        double seconds = ms / 1000.0;
        double bandwidthNaive = (2.0 * bytes) / seconds / 1e9; // GB/s

        // ----------------- Kernel 2: Shared Memory -----------------
        CUDA_CHECK(cudaEventRecord(start, 0));
        // Allocate shared memory size: threadsPerBlock * sizeof(float)
        kernel_shared<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_src, d_dst, N);
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        seconds = ms / 1000.0;
        double bandwidthShared = (2.0 * bytes) / seconds / 1e9; // GB/s

        // Print results
        printf("%10d %15.3f %15.3f\n", N, bandwidthNaive, bandwidthShared);

        // Clean up
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_src));
        CUDA_CHECK(cudaFree(d_dst));
        free(h_src);
        free(h_dst);
    }

    return 0;
}
```