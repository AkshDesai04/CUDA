```cuda
/*********************************************************************
Aim of the program:
Implement the loading phase carefully. Each thread can load one element, 
but some threads (at the edges of the block) might need to load the 
extra ghost cells.

Commentary on my thinking:
- The task is to demonstrate a correct loading of data into shared 
  memory for a stencil computation, paying special attention to 
  ghost cells that are needed for boundary threads.
- I chose a simple 1‑D stencil for clarity. Each thread loads its 
  own element into shared memory. If the thread is at the left edge 
  of the block, it also loads the element immediately to the left 
  of the block (ghost left). If it is at the right edge, it loads 
  the element immediately to the right (ghost right).
- Shared memory size is blockDim.x + 2 to accommodate the ghost 
  cells at both ends. Indexing in shared memory is offset by +1 
  for the main data, so s_mem[1] corresponds to the first element of 
  the block.
- After loading, threads synchronize, then perform a simple 
  stencil operation: each output element is the average of itself 
  and its two neighbors. Boundary elements (first and last of 
  the global array) are handled separately to avoid out‑of‑bounds.
- The code includes allocation of host and device memory, kernel 
  launch, copying back results, and basic error checking. 
- The program is self‑contained and can be compiled with nvcc.
*********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BLOCK_DIM 256

/* Kernel performing a simple 1‑D stencil with ghost cells.
   Each thread loads its element into shared memory.
   Leftmost thread also loads the left ghost cell (if any).
   Rightmost thread also loads the right ghost cell (if any). */
__global__ void stencilKernel(const float* __restrict__ d_in,
                              float* __restrict__ d_out,
                              int N)
{
    extern __shared__ float s_mem[];

    int gid = blockIdx.x * blockDim.x + threadIdx.x; // global index
    int lid = threadIdx.x;                           // local index

    /* Load the main element */
    if (gid < N)
        s_mem[lid + 1] = d_in[gid];
    else
        s_mem[lid + 1] = 0.0f;  // padding for out‑of‑range threads

    /* Load left ghost cell */
    if (lid == 0) {
        if (gid > 0)                // not the very first global element
            s_mem[0] = d_in[gid - 1];
        else
            s_mem[0] = 0.0f;        // boundary condition
    }

    /* Load right ghost cell */
    if (lid == blockDim.x - 1) {
        if (gid + 1 < N)
            s_mem[lid + 2] = d_in[gid + 1];
        else
            s_mem[lid + 2] = 0.0f;  // boundary condition
    }

    __syncthreads();

    /* Perform the stencil only for valid global indices */
    if (gid < N) {
        float left  = s_mem[lid];       // left neighbor
        float mid   = s_mem[lid + 1];   // current element
        float right = s_mem[lid + 2];   // right neighbor

        d_out[gid] = (left + mid + right) / 3.0f;
    }
}

int main(int argc, char* argv[])
{
    int N = 1024;  // size of the array
    if (argc > 1) N = atoi(argv[1]);

    size_t size_bytes = N * sizeof(float);

    /* Allocate host memory */
    float *h_in  = (float*)malloc(size_bytes);
    float *h_out = (float*)malloc(size_bytes);
    if (!h_in || !h_out) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize input array with some values */
    for (int i = 0; i < N; ++i)
        h_in[i] = (float)i;

    /* Allocate device memory */
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, size_bytes);
    cudaMalloc((void**)&d_out, size_bytes);

    /* Copy input to device */
    cudaMemcpy(d_in, h_in, size_bytes, cudaMemcpyHostToDevice);

    /* Determine grid dimensions */
    int threadsPerBlock = BLOCK_DIM;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel with shared memory of size (threadsPerBlock + 2) * sizeof(float) */
    size_t sharedMemSize = (threadsPerBlock + 2) * sizeof(float);
    stencilKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_in, d_out, N);

    /* Check for kernel launch errors */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        free(h_in);
        free(h_out);
        return EXIT_FAILURE;
    }

    /* Copy result back to host */
    cudaMemcpy(h_out, d_out, size_bytes, cudaMemcpyDeviceToHost);

    /* Simple verification: print first 10 elements */
    printf("First 10 results:\n");
    for (int i = 0; i < 10 && i < N; ++i)
        printf("%f ", h_out[i]);
    printf("\n");

    /* Clean up */
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return EXIT_SUCCESS;
}
```