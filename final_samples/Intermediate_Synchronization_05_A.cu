```cpp
/*
The kernel will take an `int* input_data` and an `int* bins`. Each thread processes one or more elements. For each element `val`, it does `atomicAdd(&bins[val], 1)`.

Thinking process:
- The goal is to count the frequency of each integer value in an input array using CUDA.
- Each element of `input_data` may have a value that serves as an index into the `bins` array.
- We will allocate a device array `bins` with a size equal to the maximum value + 1 (or a user-specified number of bins).
- The kernel will have each thread process multiple elements in a strided loop to cover the whole input array regardless of size.
- For each element read, the thread will use `atomicAdd(&bins[val], 1)` to safely increment the counter corresponding to that value.
- The host code will:
  1. Generate or load an input array of integers.
  2. Determine the number of bins needed.
  3. Allocate device memory for both input and bins.
  4. Copy data to the device.
  5. Launch the kernel with an appropriate grid/block configuration.
  6. Copy the results back and print or verify them.
- Edge cases: input values must be within the range [0, num_bins-1]; otherwise the kernel would read/write out of bounds. We assume valid data for simplicity.
- For demonstration, we will create a small input array, run the kernel, and print the resulting histogram on the host.

Note: This is a minimal, self‑contained example and omits advanced error checking or performance tuning for brevity.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

__global__
void histogramKernel(const int *input_data, int *bins, int num_elements, int num_bins)
{
    // Strided loop so each thread processes multiple elements if needed
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < num_elements; i += stride)
    {
        int val = input_data[i];
        if (val >= 0 && val < num_bins)
        {
            atomicAdd(&bins[val], 1);
        }
    }
}

int main()
{
    // Seed random number generator
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Example input: generate 1,000,000 random integers between 0 and 9
    const int num_elements = 1'000'000;
    const int num_bins = 10; // bins for values 0-9

    std::vector<int> h_input(num_elements);
    for (int i = 0; i < num_elements; ++i)
    {
        h_input[i] = std::rand() % num_bins;
    }

    // Device memory allocation
    int *d_input = nullptr;
    int *d_bins = nullptr;
    cudaMalloc(&d_input, num_elements * sizeof(int));
    cudaMalloc(&d_bins, num_bins * sizeof(int));

    // Initialize bins to zero on device
    cudaMemset(d_bins, 0, num_bins * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, h_input.data(), num_elements * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocks = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    histogramKernel<<<blocks, threadsPerBlock>>>(d_input, d_bins, num_elements, num_bins);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy results back to host
    std::vector<int> h_bins(num_bins);
    cudaMemcpy(h_bins.data(), d_bins, num_bins * sizeof(int), cudaMemcpyDeviceToHost);

    // Print histogram
    std::cout << "Histogram of values (0-" << num_bins - 1 << "):\n";
    for (int i = 0; i < num_bins; ++i)
    {
        std::cout << "Value " << i << ": " << h_bins[i] << '\n';
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_bins);

    return 0;
}
```