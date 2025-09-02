/*
Test your modified kernel with non-multiple sizes like 500x500 and verify correctness.

My thinking process for this program:

1. **Objective**  
   The aim is to demonstrate that a CUDA kernel can handle matrix dimensions that are not exact multiples of the thread block size.  
   We will use a simple operation (doubling each element) to keep the example focused on the correctness of grid/block sizing rather than the computation itself.

2. **Kernel Design**  
   - Each thread will compute its global 2‑D index (`row`, `col`).  
   - It will first check that the index is within the bounds of the matrix.  
   - If it is, it will perform the operation: `d_out[row * width + col] = d_in[row * width + col] * 2.0f;`.

3. **Grid/Block Configuration**  
   - Block size is fixed to `BLOCK_DIM_X = 16`, `BLOCK_DIM_Y = 16`.  
   - Grid size is computed as `ceil(width / BLOCK_DIM_X)` and `ceil(height / BLOCK_DIM_Y)`.  
   - This guarantees coverage of the entire matrix even when width or height is not a multiple of the block dimension.

4. **Host Code**  
   - Allocate host arrays `h_in` and `h_out_cpu` of size `width * height`.  
   - Initialize `h_in` with a simple pattern (e.g., `i + j`) so we can easily verify results.  
   - Allocate device memory for `d_in` and `d_out`.  
   - Copy input to device.  
   - Launch the kernel with the computed grid/block dimensions.  
   - Copy the result back to `h_out_gpu`.  
   - Compute the CPU reference result (`h_out_cpu`) by doubling the input.  
   - Compare `h_out_gpu` and `h_out_cpu` element‑wise and report any mismatches.  
   - Print a success/failure message.

5. **Error Checking**  
   - Wrap CUDA API calls with an error‑checking macro `CUDA_CHECK`.  
   - Handle errors by printing a message and exiting.

6. **Performance**  
   - The test is intentionally simple; the focus is correctness, not performance.

7. **Compilation**  
   - This file is a single CUDA source file (`.cu`).  
   - Compile with: `nvcc -o test_kernel test_kernel.cu`.

8. **Result**  
   - On a correct implementation, the program will print “All results match the CPU reference.”  
   - If any mismatch occurs, it will print the index and the differing values.

With this plan, the program provides a self‑contained test of a kernel’s ability to handle non‑multiple matrix sizes and verifies the result against a CPU implementation.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

/* Macro for checking CUDA errors following a CUDA API call or kernel launch */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* Kernel that doubles each element of a 2D array */
__global__ void double_kernel(const float *d_in, float *d_out, int width, int height)
{
    // Compute global indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds check
    if (col < width && row < height)
    {
        int idx = row * width + col;
        d_out[idx] = d_in[idx] * 2.0f;
    }
}

int main(void)
{
    // Matrix dimensions (non‑multiple of block dimensions)
    const int width = 500;
    const int height = 500;
    const size_t num_elements = (size_t)width * height;
    const size_t size_in_bytes = num_elements * sizeof(float);

    // Allocate host memory
    float *h_in = (float *)malloc(size_in_bytes);
    float *h_out_gpu = (float *)malloc(size_in_bytes);
    float *h_out_cpu = (float *)malloc(size_in_bytes);

    if (!h_in || !h_out_gpu || !h_out_cpu)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input with a pattern: h_in[row * width + col] = row + col
    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            h_in[row * width + col] = (float)(row + col);
        }
    }

    // Allocate device memory
    float *d_in = NULL;
    float *d_out = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_in, size_in_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, size_in_bytes));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size_in_bytes, cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim((width + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
                 (height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    // Launch kernel
    double_kernel<<<gridDim, blockDim>>>(d_in, d_out, width, height);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, size_in_bytes, cudaMemcpyDeviceToHost));

    // Compute CPU reference result
    for (int i = 0; i < num_elements; ++i)
    {
        h_out_cpu[i] = h_in[i] * 2.0f;
    }

    // Verify results
    int mismatches = 0;
    for (int i = 0; i < num_elements; ++i)
    {
        if (h_out_gpu[i] != h_out_cpu[i])
        {
            if (mismatches < 10) // Print first few mismatches
            {
                int row = i / width;
                int col = i % width;
                fprintf(stderr,
                        "Mismatch at (%d, %d): GPU=%f, CPU=%f\n",
                        row, col, h_out_gpu[i], h_out_cpu[i]);
            }
            ++mismatches;
        }
    }

    if (mismatches == 0)
    {
        printf("All results match the CPU reference.\n");
    }
    else
    {
        printf("Found %d mismatches between GPU and CPU results.\n", mismatches);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out_gpu);
    free(h_out_cpu);

    return EXIT_SUCCESS;
}
