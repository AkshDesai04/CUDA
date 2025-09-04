```cpp
/*
Aim: Implement the logic. You'll need one device buffer big enough for one chunk. 
The host loop will copy a chunk, call the kernel, and copy the result back for each chunk.

Thought process:
- The problem describes a typical GPU streaming pattern: process large data that doesn't fit in GPU memory in smaller chunks.
- We will create a simple example: multiply each element of a large array by 2. This demonstrates copying chunks to the device, launching a kernel, and copying results back.
- The host will:
    1. Allocate a host array with a large number of elements (e.g., 10 million).
    2. Allocate a device buffer that can hold a single chunk (e.g., 1 million elements).
    3. Loop over the input array in chunks:
        a. Copy a chunk from host to device using cudaMemcpy.
        b. Launch a kernel that processes that chunk.
        c. Copy the processed chunk back to the host output array.
- We'll use a simple CUDA kernel `scale_by_two` that multiplies each element by 2.
- Error checking for CUDA API calls is included.
- The program prints a message for each chunk processed and verifies the final result for a few elements.
*/

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__  \
                      << " : " << cudaGetErrorString(err) << std::endl;   \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

__global__ void scale_by_two(const float* input, float* output, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] * 2.0f;
    }
}

int main() {
    const size_t total_elements = 10'000'000;      // Total data size
    const size_t chunk_size = 1'000'000;           // Size per chunk

    // Allocate host input and output arrays
    std::vector<float> h_input(total_elements);
    std::vector<float> h_output(total_elements);

    // Initialize input data
    for (size_t i = 0; i < total_elements; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // Allocate device buffer for one chunk
    float* d_buffer_in = nullptr;
    float* d_buffer_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_buffer_in, chunk_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_buffer_out, chunk_size * sizeof(float)));

    // Determine launch configuration
    const int threads_per_block = 256;
    const int blocks_per_grid = (chunk_size + threads_per_block - 1) / threads_per_block;

    size_t processed = 0;
    while (processed < total_elements) {
        size_t current_chunk = std::min(chunk_size, total_elements - processed);

        // Copy chunk from host to device
        CHECK_CUDA(cudaMemcpy(d_buffer_in,
                              h_input.data() + processed,
                              current_chunk * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Launch kernel
        scale_by_two<<<blocks_per_grid, threads_per_block>>>(d_buffer_in,
                                                             d_buffer_out,
                                                             current_chunk);
        CHECK_CUDA(cudaGetLastError());

        // Copy result back to host
        CHECK_CUDA(cudaMemcpy(h_output.data() + processed,
                              d_buffer_out,
                              current_chunk * sizeof(float),
                              cudaMemcpyDeviceToHost));

        processed += current_chunk;
        std::cout << "Processed " << processed << " / " << total_elements << " elements.\n";
    }

    // Verify a few elements
    bool ok = true;
    for (size_t i = 0; i < 5; ++i) {
        if (h_output[i] != h_input[i] * 2.0f) {
            std::cerr << "Mismatch at index " << i << ": "
                      << h_output[i] << " != " << h_input[i] * 2.0f << std::endl;
            ok = false;
            break;
        }
    }
    if (ok) std::cout << "Verification passed for first 5 elements." << std::endl;

    // Clean up
    CHECK_CUDA(cudaFree(d_buffer_in));
    CHECK_CUDA(cudaFree(d_buffer_out));

    return 0;
}
```