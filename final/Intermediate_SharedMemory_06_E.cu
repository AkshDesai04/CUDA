/*
Extend this to a 5-point stencil.

In the original assignment the focus was on a simple stencil operation (for instance a 3‑point stencil
where each output element depended only on its left/right neighbors).  The goal here is to
generalise that logic to a true 5‑point stencil that uses the current element and its four
direct neighbors (north, south, east, west).  The 5‑point stencil is the most common
finite‑difference approximation of the 2‑D Laplacian and is frequently used in heat‑diffusion,
Poisson equation solvers, and image processing.

To achieve this we:
1. Allocate a 2‑D array on the host and initialise it with arbitrary values.
2. Allocate corresponding device memory, copy the host data over, and launch a CUDA kernel.
3. The kernel reads the element at (i, j) and its four neighbours.  For boundary points
   (i = 0, i = nx‑1, j = 0, j = ny‑1) we simply copy the input value to avoid out‑of‑bounds
   accesses; this keeps the implementation simple and mimics Dirichlet boundary
   conditions.
4. The stencil computation performed is the arithmetic mean of the five values:
      out[i,j] = (in[i,j] + in[i‑1,j] + in[i+1,j] + in[i,j‑1] + in[i,j+1]) / 5.
   (Other coefficients could be used, e.g. a Laplacian operator, but the mean is easy
   to verify and does not depend on grid spacing.)
5. Copy the results back to the host, display a few sample values, and clean up.

The code is intentionally kept simple so that it compiles with a standard CUDA toolkit
and illustrates the 5‑point stencil pattern clearly.  In a production setting one would
optimize memory access with shared memory tiling, apply multiple iterations, and use
double‑buffering, but those details are beyond the scope of this example.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Grid dimensions
#define NX 512
#define NY 512

// Kernel to perform a 5‑point stencil
__global__ void stencil5point(const float* __restrict__ in,
                              float* __restrict__ out,
                              int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (i >= nx || j >= ny) return; // out of bounds guard

    int idx = j * nx + i;

    // For boundaries we simply copy the input value
    if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        out[idx] = in[idx];
        return;
    }

    // Access the four direct neighbours
    float left   = in[idx - 1];
    float right  = in[idx + 1];
    float top    = in[idx - nx];
    float bottom = in[idx + nx];
    float center = in[idx];

    // Simple average of the five points
    out[idx] = (center + left + right + top + bottom) / 5.0f;
}

// Helper function to check CUDA errors
void checkCuda(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s (%s)\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    size_t size = NX * NY * sizeof(float);
    float *h_in  = (float*)malloc(size);
    float *h_out = (float*)malloc(size);

    // Initialize input array with some values (e.g., random numbers)
    for (int j = 0; j < NY; ++j) {
        for (int i = 0; i < NX; ++i) {
            h_in[j * NX + i] = (float)(rand()) / RAND_MAX;
        }
    }

    // Device memory allocation
    float *d_in, *d_out;
    checkCuda(cudaMalloc((void**)&d_in,  size), "alloc d_in");
    checkCuda(cudaMalloc((void**)&d_out, size), "alloc d_out");

    // Copy input to device
    checkCuda(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice), "memcpy H2D");

    // Kernel launch configuration
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    stencil5point<<<numBlocks, threadsPerBlock>>>(d_in, d_out, NX, NY);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "kernel sync");

    // Copy result back to host
    checkCuda(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost), "memcpy D2H");

    // Simple verification: print a few values
    printf("Sample output values:\n");
    for (int j = 0; j < NY; j += NY / 4) {
        for (int i = 0; i < NX; i += NX / 4) {
            printf("%0.4f ", h_out[j * NX + i]);
        }
        printf("\n");
    }

    // Clean up
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
