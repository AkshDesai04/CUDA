```cpp
/*
For the read `input[y * width + x]`, threads in a warp have the same `y` and consecutive `x`. This is coalesced.

I need to write a CUDA program that demonstrates the concept of memory coalescing when a warp of threads accesses consecutive elements of a 1D linearized 2D array. The idea is that for each warp, the `y` coordinate stays the same (threads in the warp are in the same row) and the `x` coordinate varies consecutively across the warp, resulting in contiguous memory accesses.

The program will:
1. Allocate a 2D array on the host, linearized into a 1D array.
2. Fill it with some test data.
3. Copy it to device memory.
4. Launch a kernel where each thread reads `input[y * width + x]` and writes a simple transformed value to an output array.
5. Copy the result back to the host.
6. Verify the output matches expected results.

For the kernel launch, I will use a 2D block with `blockDim.x = 32` (a warp) and `blockDim.y = 1`. Each warp will therefore contain threads with the same `y` and consecutive `x`, ensuring coalesced reads.

The code includes:
- A simple CUDA error-checking macro.
- A host `main` that sets up the data, runs the kernel, and checks results.
- The kernel `readCoalescedKernel` that performs the memory accesses.

This program is self‑contained, minimal, and illustrates the coalesced memory pattern described in the prompt.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Simple CUDA error checking macro
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel that reads from input[y * width + x] and writes a transformed value to output
__global__ void readCoalescedKernel(const float *input, float *output,
                                    int width, int height)
{
    // Compute 2D coordinates for each thread
    int x = blockIdx.x * blockDim.x + threadIdx.x; // threadIdx.x ranges 0..blockDim.x-1
    int y = blockIdx.y * blockDim.y + threadIdx.y; // threadIdx.y ranges 0..blockDim.y-1

    // Bounds check
    if (x < width && y < height) {
        // Linear index for the 2D array
        int idx = y * width + x;

        // Coalesced read: threads in the same warp have same y and consecutive x
        float val = input[idx];

        // Simple transformation: multiply by 2.0f
        output[idx] = val * 2.0f;
    }
}

int main()
{
    // Define image dimensions
    const int width  = 256;   // Must be a multiple of 32 for warp alignment
    const int height = 128;
    const int size   = width * height;

    // Allocate host memory
    float *h_input  = (float *)malloc(size * sizeof(float));
    float *h_output = (float *)malloc(size * sizeof(float));

    // Initialize input data
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            h_input[y * width + x] = static_cast<float>(y * width + x);
        }
    }

    // Allocate device memory
    float *d_input  = nullptr;
    float *d_output = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&d_input,  size * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_output, size * sizeof(float)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockDim(32, 1); // 32 threads per block along x (warp size)
    dim3 gridDim((width  + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    readCoalescedKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    CHECK_CUDA(cudaGetLastError());  // Check for launch errors

    // Copy result back to host
    CHECK_CCUDA(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few values
    bool ok = true;
    for (int y = 0; y < height && ok; ++y) {
        for (int x = 0; x < width && ok; ++x) {
            int idx = y * width + x;
            float expected = h_input[idx] * 2.0f;
            if (fabs(h_output[idx] - expected) > 1e-5f) {
                printf("Mismatch at (%d,%d): host=%f, device=%f\n",
                       x, y, expected, h_output[idx]);
                ok = false;
            }
        }
    }

    if (ok) {
        printf("Kernel executed successfully. Coalesced memory accesses were performed.\n");
    } else {
        printf("Kernel produced incorrect results.\n");
    }

    // Clean up
    free(h_input);
    free(h_output);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
```