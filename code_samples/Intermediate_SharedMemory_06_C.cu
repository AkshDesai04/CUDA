```cpp
/* 
Aim of the program:
Implement the loading phase carefully. Each thread can load one element, but some threads (at the edges of the block) might need to load the extra ghost cells.

Thinking:
The task requires a CUDA kernel that performs a 2‑D stencil on a float array.  
The loading phase is critical: each thread loads its own cell into shared memory, but when a thread is on a block boundary it must also load the adjacent ghost (halo) cells that belong to the neighboring blocks.  This ensures that during the stencil calculation every thread has access to the required neighboring values in shared memory, avoiding costly global memory accesses for every neighbor.  

To achieve this:
1. We define a tile size (TILEX × TILEY) for a thread block.  
2. Shared memory is declared with an extra border: (TILEX+2) × (TILEY+2).  
3. Each thread writes its own element into the inner part of the tile.  
4. Edge threads load the ghost cells:  
   - Left/right ghost cells for threads with `tx==0` or `tx==blockDim.x-1`.  
   - Top/bottom ghost cells for threads with `ty==0` or `ty==blockDim.y-1`.  
   Corner ghosts may be written by multiple threads; this is harmless.  
5. Bounds checking is performed to avoid reading out of range; out‑of‑bounds loads return zero.  
6. After a `__syncthreads()`, each thread performs a simple 5‑point stencil using the data in shared memory.

The host code allocates a test array, launches the kernel, and prints a few results to verify correctness.  
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILEX 16
#define TILEY 16

// Simple 5‑point stencil kernel with careful loading of ghost cells
__global__ void stencilKernel(const float* __restrict__ input,
                              float* __restrict__ output,
                              int width, int height)
{
    // Shared memory tile with halo
    extern __shared__ float shared[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int globalX = blockIdx.x * TILEX + tx;
    int globalY = blockIdx.y * TILEY + ty;

    // Index in shared memory (shifted by +1 to account for halo)
    int sharedX = tx + 1;
    int sharedY = ty + 1;
    int sharedIdx = sharedY * (TILEX + 2) + sharedX;

    // Load central element
    if (globalX < width && globalY < height)
        shared[sharedIdx] = input[globalY * width + globalX];
    else
        shared[sharedIdx] = 0.0f;

    // Load left ghost
    if (tx == 0)
    {
        int ghostX = globalX - 1;
        int ghostY = globalY;
        if (ghostX >= 0 && ghostY < height)
            shared[sharedY * (TILEX + 2) + 0] = input[ghostY * width + ghostX];
        else
            shared[sharedY * (TILEX + 2) + 0] = 0.0f;
    }

    // Load right ghost
    if (tx == TILEX - 1)
    {
        int ghostX = globalX + 1;
        int ghostY = globalY;
        if (ghostX < width && ghostY < height)
            shared[sharedY * (TILEX + 2) + (TILEX + 1)] = input[ghostY * width + ghostX];
        else
            shared[sharedY * (TILEX + 2) + (TILEX + 1)] = 0.0f;
    }

    // Load top ghost
    if (ty == 0)
    {
        int ghostX = globalX;
        int ghostY = globalY - 1;
        if (ghostX < width && ghostY >= 0)
            shared[0 * (TILEX + 2) + sharedX] = input[ghostY * width + ghostX];
        else
            shared[0 * (TILEX + 2) + sharedX] = 0.0f;
    }

    // Load bottom ghost
    if (ty == TILEY - 1)
    {
        int ghostX = globalX;
        int ghostY = globalY + 1;
        if (ghostX < width && ghostY < height)
            shared[(TILEY + 1) * (TILEX + 2) + sharedX] = input[ghostY * width + ghostX];
        else
            shared[(TILEY + 1) * (TILEX + 2) + sharedX] = 0.0f;
    }

    // Corner ghosts (optional, handled by other threads, harmless)
    __syncthreads();

    // Perform stencil only for valid output positions
    if (globalX < width && globalY < height)
    {
        float center = shared[sharedIdx];
        float left   = shared[sharedIdx - 1];
        float right  = shared[sharedIdx + 1];
        float top    = shared[sharedIdx - (TILEX + 2)];
        float bottom = shared[sharedIdx + (TILEX + 2)];

        // Simple 5‑point average
        output[globalY * width + globalX] = (center + left + right + top + bottom) * 0.2f;
    }
}

int main()
{
    const int width  = 64;
    const int height = 64;
    const int size   = width * height;
    const int bytes  = size * sizeof(float);

    // Host memory
    float *h_input  = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);

    // Initialize input with some pattern
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            h_input[y * width + x] = (float)(x + y);

    // Device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    dim3 threads(TILEX, TILEY);
    dim3 blocks((width + TILEX - 1) / TILEX, (height + TILEY - 1) / TILEY);
    size_t sharedMemSize = (TILEX + 2) * (TILEY + 2) * sizeof(float);

    // Launch kernel
    stencilKernel<<<blocks, threads, sharedMemSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // Print a few results
    printf("Sample output values:\n");
    for (int y = 0; y < 5; ++y)
    {
        for (int x = 0; x < 5; ++x)
            printf("%0.2f ", h_output[y * width + x]);
        printf("\n");
    }

    // Clean up
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
```