```cpp
/*
Profile the histogram kernel. The performance will be limited by the contention on the atomic operations.

Thinking:
- The aim is to provide a minimal CUDA program that computes a histogram of random integer data
  using a kernel that performs atomic additions on a global histogram array. 
- The kernel is intentionally simple so that the main bottleneck is contention on the atomic
  operations, as stated in the prompt. 
- To profile this kernel we use CUDA events to time the kernel launch and copy operations.
- The program will:
  1. Generate a large array of random integers on the host.
  2. Allocate device memory for the input array and the histogram (bins).
  3. Copy the input to the device.
  4. Launch the histogram kernel with a configurable grid/block size.
  5. Copy the histogram back to the host.
  6. Print timing information and a small portion of the histogram for verification.
- We include error checking after CUDA API calls and provide useful error messages.
- The user can compile this with nvcc and run it with optional command line arguments for
  number of elements and number of bins.
- The kernel uses `atomicAdd` on global memory, which will cause high contention when many
  threads try to update the same bin simultaneously. This is the main performance limitation
  we want to observe.
- No shared‑memory tiling is used, keeping the code simple and focused on atomic contention.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <cstring>

// Utility macro for CUDA error checking
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel: Naive histogram using atomicAdd on global memory
__global__ void histogram_kernel(const int *input, int *histogram,
                                 int num_elements, int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int val = input[idx];
        if (val >= 0 && val < num_bins) {
            // Contention occurs here when many threads try to update the same bin
            atomicAdd(&histogram[val], 1);
        }
    }
}

int main(int argc, char *argv[]) {
    // Default parameters
    size_t num_elements = 1 << 24; // 16 million elements
    int   num_bins     = 256;

    // Parse command line arguments (optional)
    if (argc >= 2) num_elements = std::stoul(argv[1]);
    if (argc >= 3) num_bins     = std::stoi(argv[2]);

    printf("Histogram size: %zu elements, %d bins\n", num_elements, num_bins);

    // Host memory allocation
    std::vector<int> h_input(num_elements);
    std::vector<int> h_histogram(num_bins, 0);

    // Generate random data
    srand(1234);
    for (size_t i = 0; i < num_elements; ++i) {
        h_input[i] = rand() % num_bins; // values in [0, num_bins-1]
    }

    // Device memory allocation
    int *d_input = nullptr;
    int *d_histogram = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_input, num_elements * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_histogram, num_bins * sizeof(int)));

    // Initialize histogram on device to zero
    CUDA_CHECK(cudaMemset(d_histogram, 0, num_bins * sizeof(int)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                          num_elements * sizeof(int), cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    // CUDA events for profiling
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch kernel
    histogram_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_histogram,
                                                         num_elements, num_bins);
    CUDA_CHECK(cudaGetLastError());

    // Record stop
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Compute elapsed time
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("Kernel execution time: %f ms\n", ms);

    // Copy histogram back to host
    CUDA_CHECK(cudaMemcpy(h_histogram.data(), d_histogram,
                          num_bins * sizeof(int), cudaMemcpyDeviceToHost));

    // Print a summary of the histogram
    printf("Histogram (first 10 bins):\n");
    for (int i = 0; i < 10 && i < num_bins; ++i) {
        printf("  Bin %3d: %d\n", i, h_histogram[i]);
    }

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_histogram));

    return 0;
}
```