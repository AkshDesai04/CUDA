/*
Aim of the program: Extend this to a 5-point stencil.

Thinking process:
1. The original prompt requests an extension to a 5-point stencil. A 5‑point stencil in 2‑D typically involves the center element and its four immediate neighbors (up, down, left, right).  
2. I will implement a CUDA kernel that computes a new value for each grid point as a simple average of itself and its four neighbors. This is a common pattern used for smoothing or Laplacian operators.  
3. Boundary handling: For simplicity, I’ll treat out‑of‑bounds neighbors as zero (i.e., ignore them in the sum). The kernel will check bounds before accessing neighbors.  
4. Host side: Allocate a 2‑D array (flattened to 1‑D) of size `width * height`, initialize it with some pattern (e.g., a gradient or random values). Allocate device memory, copy data, launch kernel, copy results back, and print a small portion of the array to verify correctness.  
5. Performance: Use a 2‑D block and grid configuration with block size 16×16, which is a common choice for many GPU devices.  
6. Error checking: Wrap CUDA API calls with a helper macro to check errors and abort on failure.  
7. Keep the code self‑contained in a single .cu file, as requested, with no external dependencies beyond the standard CUDA runtime.

The resulting code below demonstrates a complete CUDA C program that performs a 5‑point stencil on a 2‑D array.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__global__ void stencil5point(const float *input, float *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    float center = input[idx];
    float sum = center;
    int count = 1; // start with the center

    // Left neighbor
    if (x > 0) {
        sum += input[idx - 1];
        count++;
    }
    // Right neighbor
    if (x < width - 1) {
        sum += input[idx + 1];
        count++;
    }
    // Top neighbor
    if (y > 0) {
        sum += input[idx - width];
        count++;
    }
    // Bottom neighbor
    if (y < height - 1) {
        sum += input[idx + width];
        count++;
    }

    // Simple average of available points
    output[idx] = sum / count;
}

int main(void)
{
    const int width  = 512; // change as desired
    const int height = 512; // change as desired
    const size_t size = width * height * sizeof(float);

    // Allocate host memory
    float *h_input  = (float *)malloc(size);
    float *h_output = (float *)malloc(size);
    if (!h_input || !h_output) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input with a simple pattern (e.g., a gradient)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            h_input[y * width + x] = (float)(x + y);
        }
    }

    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc((void **)&d_input,  size));
    CHECK_CUDA(cudaMalloc((void **)&d_output, size));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Define block and grid sizes
    dim3 blockDim(16, 16);
    dim3 gridDim((width  + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    stencil5point<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // Print a small portion of the output for verification
    printf("Sample output (center region):\n");
    for (int y = height / 2 - 2; y <= height / 2 + 2; ++y) {
        for (int x = width  / 2 - 2; x <= width  / 2 + 2; ++x) {
            printf("%6.2f ", h_output[y * width + x]);
        }
        printf("\n");
    }

    // Clean up
    free(h_input);
    free(h_output);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return EXIT_SUCCESS;
}
