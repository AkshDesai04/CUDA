/*
To fix bank conflicts in a 2D array `shared_mem[ROW][COL]`, padding can be added: `__shared__ float tile[TILE_SIZE][TILE_SIZE+1];`. Explain how accessing `tile[threadIdx.y][threadIdx.x]` could cause conflicts, but accessing `tile[threadIdx.x][threadIdx.y]` is fine.

I thought about how CUDA shared memory is organized into 32 banks. The address of an element `tile[i][j]` is calculated as `base + (i * stride + j) * sizeof(float)`.  
When `stride` equals 32 (e.g. `tile[TILE][TILE]` with `TILE == 32`) all elements in the same row are separated by exactly 32 floats. For a warp of 32 threads that all have the same `i` but different `j`, the linear addresses differ by one float each, so the bank index is `(i*32 + j) mod 32 = j mod 32`.  
Because each thread in the warp accesses a different bank, this is *not* a conflict.

However, when threads in the same warp have different `i` values (e.g. a 2‑D block where each warp covers several rows), the bank index becomes `(i*32 + j) mod 32`.  
If `i` changes by 1 for consecutive threads, the bank index stays the same, causing all those threads to access the same bank simultaneously – a **bank conflict**.  
This situation arises when we load data into `tile[threadIdx.y][threadIdx.x]` in a naive transpose kernel: many threads in the same warp may share the same `threadIdx.x` (column) but differ in `threadIdx.y` (row), making the stride 32 align multiple accesses to the same bank.

Padding the second dimension (`TILE_SIZE+1`) changes the stride to 33, an odd number. Now the bank index for the same pattern becomes `(i*33 + j) mod 32`. Because 33 ≡ 1 (mod 32), the bank index increments by 1 for each row, so consecutive threads no longer collide.  
Additionally, if we swap the indices when accessing shared memory (i.e., use `tile[threadIdx.x][threadIdx.y]`), the effective stride becomes 32 again, but now the varying dimension is the column index. Since in a warp the column index changes for every thread and the row index is constant, the addresses map to different banks, eliminating the conflict.

The following CUDA program demonstrates a simple matrix transpose. It includes:
* a naive kernel that uses `tile[threadIdx.y][threadIdx.x]` and suffers from bank conflicts,
* an optimized kernel that pads the shared array and accesses it as `tile[threadIdx.x][threadIdx.y]` to avoid conflicts.
The program runs both kernels on a small matrix, prints the results, and shows that the optimized version works correctly without the bank‑conflict issue.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE 32
#define WIDTH  64
#define HEIGHT 64

// Naive transpose kernel – may suffer from bank conflicts
__global__ void transpose_naive(const float* __restrict__ A, float* __restrict__ B, int width, int height)
{
    __shared__ float tile[TILE][TILE];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < width && y < height)
    {
        // Potential bank conflict when reading from global memory into shared memory
        tile[threadIdx.y][threadIdx.x] = A[y * width + x];
    }
    __syncthreads();

    if (x < width && y < height)
    {
        // Transpose by swapping indices when writing back
        B[x * height + y] = tile[threadIdx.x][threadIdx.y];
    }
}

// Optimized transpose kernel with padding – eliminates bank conflicts
__global__ void transpose_padded(const float* __restrict__ A, float* __restrict__ B, int width, int height)
{
    __shared__ float tile[TILE][TILE + 1]; // Padding to avoid conflicts

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < width && y < height)
    {
        // No bank conflict because stride is TILE+1
        tile[threadIdx.y][threadIdx.x] = A[y * width + x];
    }
    __syncthreads();

    if (x < width && y < height)
    {
        // Safe access with no conflict
        B[x * height + y] = tile[threadIdx.x][threadIdx.y];
    }
}

void init_matrix(float* mat, int width, int height)
{
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            mat[i * width + j] = static_cast<float>(i * width + j);
}

void print_matrix(const float* mat, int width, int height)
{
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
            printf("%6.1f ", mat[i * width + j]);
        printf("\n");
    }
    printf("\n");
}

int main()
{
    size_t size = WIDTH * HEIGHT * sizeof(float);

    float* h_A = (float*)malloc(size);
    float* h_B_naive = (float*)malloc(size);
    float* h_B_padded = (float*)malloc(size);

    init_matrix(h_A, WIDTH, HEIGHT);

    float *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    // Run naive kernel
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    dim3 block(TILE, TILE);
    dim3 grid((WIDTH + TILE - 1) / TILE, (HEIGHT + TILE - 1) / TILE);
    transpose_naive<<<grid, block>>>(d_A, d_B, WIDTH, HEIGHT);
    cudaMemcpy(h_B_naive, d_B, size, cudaMemcpyDeviceToHost);

    // Run padded kernel
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    transpose_padded<<<grid, block>>>(d_A, d_B, WIDTH, HEIGHT);
    cudaMemcpy(h_B_padded, d_B, size, cudaMemcpyDeviceToHost);

    printf("Original Matrix (first 8x8 block):\n");
    print_matrix(h_A, WIDTH, HEIGHT);

    printf("Transposed by Naive Kernel (first 8x8 block):\n");
    print_matrix(h_B_naive, HEIGHT, WIDTH);

    printf("Transposed by Padded Kernel (first 8x8 block):\n");
    print_matrix(h_B_padded, HEIGHT, WIDTH);

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B_naive);
    free(h_B_padded);

    return 0;
}
