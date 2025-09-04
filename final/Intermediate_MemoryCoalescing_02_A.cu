/*
Aim of the program:
Phase 1: Coalesced read from `input`. Thread `(tx,ty)` in a block reads from `input` at a location corresponding to its `(global_x, global_y)` and stores it in `tile[ty][tx]`.

Thinking:
The task is to demonstrate coalesced global memory reads in CUDA. Each thread in a block will load a single element from a global input array into a two‑dimensional shared memory tile. The mapping from thread coordinates `(tx, ty)` to global coordinates `(global_x, global_y)` is straightforward:  
```
global_x = blockIdx.x * blockDim.x + tx
global_y = blockIdx.y * blockDim.y + ty
```  
Since the input array is stored in row‑major order (`input[row * width + col]`), consecutive threads in the x‑dimension will access consecutive memory locations. This guarantees a fully coalesced read for a 32‑thread warp.

After loading the tile, the kernel writes the same value back to an output array to show that the load succeeded and to allow a simple correctness check. In a real application, further processing would be performed in shared memory before writing back.

The host code sets up a small example (e.g., 256×256) and verifies that the output matches the input. Basic CUDA error checking is included. No external libraries are required beyond the CUDA runtime.

The final .cu file is self‑contained, compilable, and demonstrates the described coalesced read pattern.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16  // 16x16 tile

// Kernel performing coalesced read into shared memory
__global__ void readCoalesced(const float* __restrict__ input,
                              float* __restrict__ output,
                              int width, int height)
{
    // 2D thread indices within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 2D block indices in grid
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Compute global indices
    int global_x = bx * blockDim.x + tx;
    int global_y = by * blockDim.y + ty;

    // Shared memory tile
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];

    // Bounds check (in case grid size not multiple of block size)
    if (global_x < width && global_y < height)
    {
        // Load from global memory into shared memory
        tile[ty][tx] = input[global_y * width + global_x];

        // Optional: write back to global output (for verification)
        output[global_y * width + global_x] = tile[ty][tx];
    }
}

int checkCUDA(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s : %s\n", msg, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int main()
{
    const int width = 256;
    const int height = 256;
    const int size = width * height;
    const size_t bytes = size * sizeof(float);

    // Allocate host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);

    // Initialize input
    for (int i = 0; i < size; ++i)
        h_input[i] = (float)i;

    // Allocate device memory
    float *d_input, *d_output;
    checkCUDA(cudaMalloc((void**)&d_input, bytes), "cudaMalloc d_input");
    checkCUDA(cudaMalloc((void**)&d_output, bytes), "cudaMalloc d_output");

    // Copy input to device
    checkCUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    // Define grid and block dimensions
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    readCoalesced<<<grid, block>>>(d_input, d_output, width, height);
    checkCUDA(cudaGetLastError(), "Kernel launch");

    // Copy output back to host
    checkCUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    // Verify results
    int errors = 0;
    for (int i = 0; i < size; ++i)
    {
        if (h_input[i] != h_output[i])
        {
            if (errors < 10)  // print first few errors
                fprintf(stderr, "Mismatch at %d: input=%f, output=%f\n", i, h_input[i], h_output[i]);
            errors++;
        }
    }
    if (errors == 0)
        printf("Verification PASSED.\n");
    else
        printf("Verification FAILED with %d mismatches.\n", errors);

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
