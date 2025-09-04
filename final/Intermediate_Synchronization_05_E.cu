```cuda
/*
Verify the GPU histogram result against a simple CPU implementation.

The aim of this program is to provide a minimal yet complete example that
computes a histogram of an array of 8‑bit values on both the CPU and the
GPU, then compares the two results to verify correctness.  
The program follows these steps:

1. Generate a host array of random unsigned 8‑bit integers.
2. Compute the histogram on the CPU by a straightforward serial loop.
3. Allocate device memory and copy the input array to the GPU.
4. Launch a CUDA kernel that builds the histogram in parallel.
   * Each thread block uses shared memory to accumulate partial histograms
     for the values 0–255.
   * After processing its portion of the input, a block writes its partial
     histogram to a global buffer.
   * A second kernel (or a simple atomicAdd loop in the first kernel)
     aggregates all block histograms into the final histogram.
5. Copy the GPU histogram back to the host.
6. Compare the CPU and GPU histograms element‑wise; report success or
   mismatches.
7. Clean up all allocated resources.

The program uses error‑checking macros for CUDA API calls and kernel
launches.  It is self‑contained and compiles with `nvcc` as a single `.cu`
file.  No external libraries are required beyond the CUDA runtime.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define HISTOGRAM_BINS 256
#define THREADS_PER_BLOCK 256
#define NUM_ELEMENTS (1 << 20)   // 1M elements

/* Error checking macro */
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

/* GPU kernel: each block computes a partial histogram in shared memory,
   then writes it to a global buffer. */
__global__ void histogram_kernel(const unsigned char* d_in,
                                 unsigned int* d_partial_histograms,
                                 size_t N)
{
    __shared__ unsigned int hist[HISTOGRAM_BINS];

    // Initialize shared histogram to zero
    for (int i = threadIdx.x; i < HISTOGRAM_BINS; i += blockDim.x) {
        hist[i] = 0;
    }
    __syncthreads();

    // Process elements assigned to this thread
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (; idx < N; idx += stride) {
        unsigned char val = d_in[idx];
        atomicAdd(&(hist[val]), 1);
    }
    __syncthreads();

    // Write block's partial histogram to global memory
    for (int i = threadIdx.x; i < HISTOGRAM_BINS; i += blockDim.x) {
        d_partial_histograms[blockIdx.x * HISTOGRAM_BINS + i] = hist[i];
    }
}

/* GPU kernel: reduce all block histograms into the final histogram. */
__global__ void reduce_histograms(const unsigned int* d_partial_histograms,
                                  unsigned int* d_final_histogram,
                                  int num_blocks)
{
    int bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin >= HISTOGRAM_BINS) return;

    unsigned int sum = 0;
    for (int i = 0; i < num_blocks; ++i) {
        sum += d_partial_histograms[i * HISTOGRAM_BINS + bin];
    }
    d_final_histogram[bin] = sum;
}

/* CPU implementation of histogram for verification */
void histogram_cpu(const unsigned char* h_in, unsigned int* h_hist, size_t N)
{
    for (size_t i = 0; i < N; ++i) {
        h_hist[h_in[i]]++;
    }
}

int main()
{
    size_t N = NUM_ELEMENTS;
    size_t size_bytes = N * sizeof(unsigned char);

    /* Allocate and initialize host input array */
    unsigned char* h_in = (unsigned char*)malloc(size_bytes);
    if (!h_in) {
        fprintf(stderr, "Failed to allocate host input array.\n");
        return EXIT_FAILURE;
    }
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < N; ++i) {
        h_in[i] = rand() % HISTOGRAM_BINS;
    }

    /* CPU histogram */
    unsigned int* h_hist_cpu = (unsigned int*)calloc(HISTOGRAM_BINS, sizeof(unsigned int));
    if (!h_hist_cpu) {
        fprintf(stderr, "Failed to allocate CPU histogram.\n");
        free(h_in);
        return EXIT_FAILURE;
    }
    histogram_cpu(h_in, h_hist_cpu, N);

    /* Allocate device memory */
    unsigned char* d_in = nullptr;
    unsigned int* d_partial_histograms = nullptr;
    unsigned int* d_final_histogram = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in, size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_partial_histograms,
                          sizeof(unsigned int) * HISTOGRAM_BINS * ((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)));
    CUDA_CHECK(cudaMalloc((void**)&d_final_histogram,
                          sizeof(unsigned int) * HISTOGRAM_BINS));

    /* Copy input to device */
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size_bytes, cudaMemcpyHostToDevice));

    /* Launch histogram kernel */
    int num_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    histogram_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_in, d_partial_histograms, N);
    CUDA_CHECK(cudaGetLastError());

    /* Reduce partial histograms */
    int reduce_blocks = (HISTOGRAM_BINS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    reduce_histograms<<<reduce_blocks, THREADS_PER_BLOCK>>>(d_partial_histograms,
                                                           d_final_histogram,
                                                           num_blocks);
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    unsigned int* h_hist_gpu = (unsigned int*)malloc(HISTOGRAM_BINS * sizeof(unsigned int));
    if (!h_hist_gpu) {
        fprintf(stderr, "Failed to allocate GPU histogram buffer.\n");
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_partial_histograms));
        CUDA_CHECK(cudaFree(d_final_histogram));
        free(h_in);
        free(h_hist_cpu);
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaMemcpy(h_hist_gpu, d_final_histogram,
                          HISTOGRAM_BINS * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    /* Verify results */
    int mismatches = 0;
    for (int i = 0; i < HISTOGRAM_BINS; ++i) {
        if (h_hist_cpu[i] != h_hist_gpu[i]) {
            mismatches++;
            printf("Mismatch at bin %d: CPU=%u GPU=%u\n",
                   i, h_hist_cpu[i], h_hist_gpu[i]);
        }
    }
    if (mismatches == 0) {
        printf("Verification PASSED: CPU and GPU histograms match.\n");
    } else {
        printf("Verification FAILED: %d mismatches found.\n", mismatches);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_partial_histograms));
    CUDA_CHECK(cudaFree(d_final_histogram));
    free(h_in);
    free(h_hist_cpu);
    free(h_hist_gpu);

    return 0;
}
```