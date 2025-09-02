```cpp
/*
 * Aim: Verify the GPU histogram result against a simple CPU implementation.
 *
 * Thinking:
 * 1. The goal is to create a small CUDA program that generates random data,
 *    computes a histogram of that data on the GPU, computes the same histogram
 *    on the CPU, and then compares the two results for correctness.
 *
 * 2. We will use 256 bins for simplicity, assuming the data values are
 *    unsigned 8‑bit integers (0–255).  The CPU histogram will be computed
 *    with a straightforward loop over all elements.
 *
 * 3. For the GPU histogram we will:
 *    - Allocate device memory for the input data and for the histogram.
 *    - Launch a kernel with a configurable number of blocks and threads.
 *    - Each thread will process one or more elements of the input array
 *      (by stride).  When it encounters a value, it will atomically
 *      increment the corresponding bin in the global histogram array.
 *
 * 4. We will use `atomicAdd` on the global histogram because the input
 *    size may be large and simple per‑thread private histograms with
 *    reductions would add complexity that is not needed for this
 *    verification test.
 *
 * 5. After kernel execution, we copy the GPU histogram back to host
 *    memory, then compare element‑by‑element with the CPU histogram.
 *
 * 6. We print a success message if all bins match; otherwise we print
 *    the first mismatched bin and the values on each side.
 *
 * 7. Basic error checking for CUDA API calls will be included using a
 *    helper macro.
 *
 * 8. The code is fully self‑contained: it includes random data generation,
 *    CPU and GPU histogram functions, comparison logic, and simple timing
 *    for each part.
 *
 * 9. The program is written in C/C++ (C99/++11) style, compiled with
 *    `nvcc` targeting CUDA, and will produce a .cu file that can be
 *    compiled directly.
 */

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",           \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

const int NUM_BINS = 256;

// Kernel: each thread processes a strided range of input elements
__global__ void histogram_kernel(const unsigned char *d_data, int N,
                                 unsigned int *d_hist)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        unsigned char val = d_data[i];
        atomicAdd(&d_hist[(int)val], 1);
    }
}

// CPU histogram: simple single‑threaded implementation
void histogram_cpu(const unsigned char *h_data, int N,
                   unsigned int *h_hist)
{
    for (int i = 0; i < NUM_BINS; ++i) {
        h_hist[i] = 0;
    }
    for (int i = 0; i < N; ++i) {
        unsigned char val = h_data[i];
        h_hist[(int)val]++;
    }
}

// Compare two histograms
bool compare_histograms(const unsigned int *h_cpu, const unsigned int *h_gpu,
                        int bins)
{
    for (int i = 0; i < bins; ++i) {
        if (h_cpu[i] != h_gpu[i]) {
            printf("Mismatch at bin %d: CPU=%u, GPU=%u\n",
                   i, h_cpu[i], h_gpu[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    const int N = 1 << 20; // 1 million elements
    unsigned char *h_data = (unsigned char*)malloc(N * sizeof(unsigned char));
    if (!h_data) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Seed RNG and fill data with random 0–255 values
    std::srand((unsigned)std::time(nullptr));
    for (int i = 0; i < N; ++i) {
        h_data[i] = (unsigned char)(std::rand() % NUM_BINS);
    }

    // Allocate device memory
    unsigned char *d_data = nullptr;
    unsigned int *d_hist = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_data, N * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc((void**)&d_hist, NUM_BINS * sizeof(unsigned int)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));

    // Initialize device histogram to zero
    CHECK_CUDA(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    histogram_kernel<<<blocks, threadsPerBlock>>>(d_data, N, d_hist);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back GPU histogram
    unsigned int *h_gpu_hist = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
    if (!h_gpu_hist) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaMemcpy(h_gpu_hist, d_hist, NUM_BINS * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    // Compute CPU histogram
    unsigned int *h_cpu_hist = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
    if (!h_cpu_hist) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }
    histogram_cpu(h_data, N, h_cpu_hist);

    // Compare results
    bool ok = compare_histograms(h_cpu_hist, h_gpu_hist, NUM_BINS);
    if (ok) {
        printf("Success: GPU histogram matches CPU histogram.\n");
    } else {
        printf("Failure: GPU histogram does not match CPU histogram.\n");
    }

    // Clean up
    free(h_data);
    free(h_gpu_hist);
    free(h_cpu_hist);
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_hist));

    return 0;
}
```