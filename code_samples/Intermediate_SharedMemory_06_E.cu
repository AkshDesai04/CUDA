```cuda
/*
Aim of the program: Extend this to a 5-point stencil.

Thinking:
The task is to implement a 5‑point stencil on a 2‑D array using CUDA.
A 5‑point stencil updates each interior grid point to the weighted average of
the point itself and its four immediate neighbors (up, down, left, right).
We will:

1. Allocate host and device memory for an N×N grid.
2. Initialize the host grid with some sample data.
3. Copy the data to the device.
4. Launch a CUDA kernel that performs the stencil once.
   The kernel will use shared memory to reduce global memory traffic.
   Boundary points will be left unchanged to keep the implementation simple.
5. Copy the result back to the host and optionally verify the result.
6. Measure performance using CUDA events.

The program uses double precision for numerical stability, but the logic
could be adapted to float if desired. The stencil coefficients are
uniformly 0.2 for the point itself and each neighbor, which sums to 1.0.

The code is self‑contained and can be compiled with:
    nvcc -O3 -std=c++11 stencil5.cu -o stencil5
and run on a system with CUDA support.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define IDX(i, j, N) ((i)*(N)+(j))
#define TILE_DIM 32
#define BLOCK_ROWS 8

// Kernel that performs a single 5‑point stencil iteration.
__global__ void stencil5_kernel(const double* __restrict__ d_in,
                                double* __restrict__ d_out,
                                int N)
{
    // Shared memory tile with halo
    __shared__ double tile[TILE_DIM+2][TILE_DIM+2];

    // Global indices of the thread
    int global_row = blockIdx.y * TILE_DIM + threadIdx.y;
    int global_col = blockIdx.x * TILE_DIM + threadIdx.x;

    // Index into the shared memory tile
    int local_row = threadIdx.y + 1; // offset for halo
    int local_col = threadIdx.x + 1;

    // Load center data
    if (global_row < N && global_col < N)
        tile[local_row][local_col] = d_in[IDX(global_row, global_col, N)];
    else
        tile[local_row][local_col] = 0.0;

    // Load halos
    // Left halo
    if (threadIdx.x == 0) {
        int col = global_col - 1;
        if (col >= 0 && global_row < N)
            tile[local_row][0] = d_in[IDX(global_row, col, N)];
        else
            tile[local_row][0] = 0.0;
    }
    // Right halo
    if (threadIdx.x == TILE_DIM - 1) {
        int col = global_col + 1;
        if (col < N && global_row < N)
            tile[local_row][TILE_DIM+1] = d_in[IDX(global_row, col, N)];
        else
            tile[local_row][TILE_DIM+1] = 0.0;
    }
    // Top halo
    if (threadIdx.y == 0) {
        int row = global_row - 1;
        if (row >= 0 && global_col < N)
            tile[0][local_col] = d_in[IDX(row, global_col, N)];
        else
            tile[0][local_col] = 0.0;
    }
    // Bottom halo
    if (threadIdx.y == TILE_DIM - 1) {
        int row = global_row + 1;
        if (row < N && global_col < N)
            tile[TILE_DIM+1][local_col] = d_in[IDX(row, global_col, N)];
        else
            tile[TILE_DIM+1][local_col] = 0.0;
    }
    // Corners (optional, not needed for stencil but set to 0)
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        tile[0][0] = 0.0;
    }
    if (threadIdx.x == TILE_DIM-1 && threadIdx.y == 0) {
        tile[0][TILE_DIM+1] = 0.0;
    }
    if (threadIdx.x == 0 && threadIdx.y == TILE_DIM-1) {
        tile[TILE_DIM+1][0] = 0.0;
    }
    if (threadIdx.x == TILE_DIM-1 && threadIdx.y == TILE_DIM-1) {
        tile[TILE_DIM+1][TILE_DIM+1] = 0.0;
    }

    __syncthreads();

    // Perform stencil if inside bounds and not on global boundary
    if (global_row < N && global_col < N &&
        global_row > 0 && global_row < N-1 &&
        global_col > 0 && global_col < N-1) {
        double center = tile[local_row][local_col];
        double north  = tile[local_row-1][local_col];
        double south  = tile[local_row+1][local_col];
        double west   = tile[local_row][local_col-1];
        double east   = tile[local_row][local_col+1];

        d_out[IDX(global_row, global_col, N)] =
            0.2 * (center + north + south + west + east);
    }
    // Optional: copy boundaries unchanged
    else if (global_row < N && global_col < N) {
        d_out[IDX(global_row, global_col, N)] =
            d_in[IDX(global_row, global_col, N)];
    }
}

int main(int argc, char* argv[])
{
    int N = 1024; // Default grid size
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            fprintf(stderr, "Invalid grid size. Using default 1024.\n");
            N = 1024;
        }
    }

    size_t size = N * N * sizeof(double);
    double *h_in = (double*)malloc(size);
    double *h_out = (double*)malloc(size);
    if (!h_in || !h_out) {
        fprintf(stderr, "Host allocation failed.\n");
        return EXIT_FAILURE;
    }

    // Initialize input grid with a simple pattern
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_in[IDX(i, j, N)] = sin(i * 0.01) * cos(j * 0.01);
        }
    }

    double *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((N + TILE_DIM - 1) / TILE_DIM,
                 (N + TILE_DIM - 1) / TILE_DIM);

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Launch kernel
    stencil5_kernel<<<gridDim, blockDim>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    printf("Stencil5 on %dx%d grid took %f ms.\n", N, N, ms);

    // Simple validation: print center value
    printf("Center value after stencil: %f\n", h_out[IDX(N/2, N/2, N)]);

    // Clean up
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```