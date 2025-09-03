```cuda
/*********************************************************************
 * Aim of the program (as requested in the prompt, word for word):
 *
 * Implement a parallel histogram calculation for an array of integers.
 * Multiple threads will process the input array and may try to increment
 * the same histogram bin simultaneously. Use atomicAdd on the bin
 * counters to prevent data races and ensure a correct histogram.
 *
 * -------------------------------------------------------------------
 * Entire thinking behind this implementation:
 *
 * 1. **Problem domain**  
 *    We want to count how many times each integer value occurs in a
 *    large array.  In a single‑threaded version this is simply a
 *    loop over the array and an increment of a bin.  In a GPU the
 *    array is split across thousands of threads.  Two or more
 *    threads might simultaneously try to increment the same bin, so
 *    we must serialize those increments to avoid lost updates.
 *
 * 2. **Atomic operations**  
 *    CUDA provides `atomicAdd` for integer types.  Using
 *    `atomicAdd(&hist[bin], 1)` guarantees that each increment
 *    completes before the next one is applied, even if many threads
 *    target the same bin.  The cost is that the operation is serialized
 *    across threads that touch the same memory address, but for a
 *    histogram this is usually acceptable.
 *
 * 3. **Kernel design**  
 *    - Each thread processes several array elements using a *grid*
 *      striding loop: `for (int i = idx; i < N; i += stride)` where
 *      `stride = gridDim.x * blockDim.x`.  
 *    - The value of the element is used as an index into the
 *      histogram array.  We assume all values lie in the range
 *      `[0, NUM_BINS-1]`.  If a value were out of range the kernel
 *      would ignore it (or could raise an error in a more robust
 *      implementation).
 *    - For each element we call `atomicAdd(&d_hist[val], 1)` to
 *      safely increment the bin.
 *
 * 4. **Memory allocation**  
 *    - Host array `h_data` of size `N` is allocated and initialized
 *      with random integers.  
 *    - Device array `d_data` holds the input.  
 *    - Device array `d_hist` holds the histogram (size `NUM_BINS`).
 *    - The histogram is initialized to zero on the device using
 *      `cudaMemset`.
 *
 * 5. **Kernel launch configuration**  
 *    - `BLOCK_SIZE` is set to 256, a common choice that yields
 *      good occupancy on many GPUs.  
 *    - `gridSize` is calculated as `(N + BLOCK_SIZE - 1) / BLOCK_SIZE`
 *      to cover all elements.
 *
 * 6. **Error checking**  
 *    - A macro `CUDA_CHECK` wraps CUDA API calls to report errors
 *      immediately.
 *
 * 7. **Verification**  
 *    - After the kernel completes, the histogram is copied back to
 *      the host and printed.  
 *    - Optionally, the host can compute a reference histogram
 *      sequentially to verify correctness.
 *
 * 8. **Portability**  
 *    - The program is self‑contained and can be compiled with
 *      `nvcc` on any CUDA‑capable system.  
 *    - It uses only basic CUDA runtime API and standard C library
 *      functions.
 *********************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_BINS 256          // Histogram bins (0..255)
#define BLOCK_SIZE 256        // Threads per block
#define ARRAY_SIZE 1048576    // Size of input array (1M elements)

// Error checking macro
#define CUDA_CHECK(call)                                        \
    do {                                                        \
        cudaError_t err = call;                                 \
        if (err != cudaSuccess) {                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while (0)

// Kernel: Each thread processes multiple elements and atomically
// increments the appropriate histogram bin.
__global__ void histogramKernel(const int *d_input, int *d_hist, size_t N)
{
    // Global thread index
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    // Stride over the input array
    for (size_t i = idx; i < N; i += stride)
    {
        int val = d_input[i];
        if (val >= 0 && val < NUM_BINS)
        {
            atomicAdd(&d_hist[val], 1);
        }
    }
}

// Host function to compute a reference histogram (sequentially)
void computeReference(const int *h_input, int *h_ref, size_t N)
{
    for (size_t i = 0; i < N; ++i)
    {
        int val = h_input[i];
        if (val >= 0 && val < NUM_BINS)
            h_ref[val] += 1;
    }
}

int main(void)
{
    // Allocate host memory
    int *h_data = (int *)malloc(ARRAY_SIZE * sizeof(int));
    int *h_hist_gpu = (int *)malloc(NUM_BINS * sizeof(int));
    int *h_hist_ref = (int *)malloc(NUM_BINS * sizeof(int));

    // Initialize input with random values in [0, NUM_BINS-1]
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < ARRAY_SIZE; ++i)
        h_data[i] = rand() % NUM_BINS;

    // Allocate device memory
    int *d_data = NULL;
    int *d_hist = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_data, ARRAY_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_hist, NUM_BINS * sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data, ARRAY_SIZE * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Initialize device histogram to zero
    CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(int)));

    // Launch kernel
    int gridSize = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    histogramKernel<<<gridSize, BLOCK_SIZE>>>(d_data, d_hist, ARRAY_SIZE);
    CUDA_CHECK(cudaGetLastError());      // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for completion

    // Copy histogram back to host
    CUDA_CHECK(cudaMemcpy(h_hist_gpu, d_hist, NUM_BINS * sizeof(int),
                          cudaMemcpyDeviceToHost));

    // Compute reference histogram on host for verification
    for (int i = 0; i < NUM_BINS; ++i) h_hist_ref[i] = 0;
    computeReference(h_data, h_hist_ref, ARRAY_SIZE);

    // Verify correctness
    int mismatches = 0;
    for (int i = 0; i < NUM_BINS; ++i)
    {
        if (h_hist_gpu[i] != h_hist_ref[i])
        {
            mismatches++;
            printf("Bin %d: GPU=%d, CPU=%d\n", i, h_hist_gpu[i], h_hist_ref[i]);
        }
    }

    if (mismatches == 0)
        printf("Histogram verified successfully! All bins match.\n");
    else
        printf("Histogram verification failed: %d mismatches.\n", mismatches);

    // Print first few bins as a quick sanity check
    printf("\nFirst 16 histogram bins (value : count):\n");
    for (int i = 0; i < 16; ++i)
        printf("%2d : %6d\n", i, h_hist_gpu[i]);

    // Clean up
    free(h_data);
    free(h_hist_gpu);
    free(h_hist_ref);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_hist));

    return 0;
}
```