```cuda
/*
Aim of the program: For the read `input[y * width + x]`, threads in a warp have the same `y` and consecutive `x`. This is coalesced.

Thinking:
1. The goal is to demonstrate how CUDA memory coalescing works when each warp accesses consecutive addresses within the same row of a 2D array.  
2. We set up a simple 2D array (`input`) on the host and copy it to device memory.  
3. A kernel (`copyKernel`) is launched with `blockDim.x` equal to 32 (the warp size).  
4. Each thread calculates its global coordinates `(x, y)` and reads from `input[y * width + x]` and writes to `output[y * width + x]`.  
5. Because all threads in a warp share the same `y` and have consecutive `x`, their memory accesses are contiguous and hence fully coalesced.  
6. After the kernel execution, we copy the result back to the host and verify that the copy was successful.  
7. The program is self‑contained and can be compiled with `nvcc`.  
8. No additional files or external dependencies are required.  
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH  256   // Number of columns
#define HEIGHT 64    // Number of rows

// Simple kernel that copies input to output.
// Each warp accesses consecutive memory addresses in the same row.
__global__ void copyKernel(const int* __restrict__ input, int* __restrict__ output, int width, int height)
{
    // Compute global x and y indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds check
    if (x < width && y < height)
    {
        int idx = y * width + x;
        output[idx] = input[idx];
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

int main(void)
{
    size_t numElements = WIDTH * HEIGHT;
    size_t sizeInBytes = numElements * sizeof(int);

    // Allocate host memory
    int *h_input  = (int*)malloc(sizeInBytes);
    int *h_output = (int*)malloc(sizeInBytes);

    if (!h_input || !h_output) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host input with some pattern
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            h_input[y * WIDTH + x] = y * WIDTH + x;  // Simple linear index
        }
    }

    // Allocate device memory
    int *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_input,  sizeInBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, sizeInBytes));

    // Copy input from host to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeInBytes, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockDim(32, 1);                 // 32 threads per block in x (warp size)
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x,
                 (HEIGHT + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    copyKernel<<<gridDim, blockDim>>>(d_input, d_output, WIDTH, HEIGHT);
    CUDA_CHECK(cudaGetLastError());  // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeInBytes, cudaMemcpyDeviceToHost));

    // Verify the result
    int errors = 0;
    for (int i = 0; i < numElements; ++i) {
        if (h_output[i] != h_input[i]) {
            if (errors < 10) {  // Print first few errors
                fprintf(stderr, "Mismatch at index %d: host=%d, device=%d\n",
                        i, h_input[i], h_output[i]);
            }
            ++errors;
        }
    }
    if (errors == 0) {
        printf("Success! All elements match.\n");
    } else {
        printf("Total mismatches: %d\n", errors);
    }

    // Clean up
    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return EXIT_SUCCESS;
}
```