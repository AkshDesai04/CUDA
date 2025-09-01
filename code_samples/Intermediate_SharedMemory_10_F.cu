/*
Aim: Use padding in the shared memory array (__shared__ float s_tile[TILE_SIZE][TILE_SIZE+1]) to eliminate bank conflicts even with non-optimal access patterns.

Thinking:
- Matrix transposition is a classic example where shared memory can be used to reduce global memory traffic.
- Without padding, each row of the shared array lies in a single memory bank, so transposed accesses cause bank conflicts.
- By padding each row with an extra column (TILE_SIZE+1), we ensure that consecutive elements in a row fall into different banks.
- The kernel reads an N×N matrix from global memory into shared memory, synchronizes, then writes the transposed result back.
- The host code creates a random matrix, copies it to the GPU, runs the transpose kernel, copies the result back, and verifies correctness.
- Timing is measured with cudaEvent_t for demonstration purposes.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define TILE_SIZE 32
#define CHECK_CUDA(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

__global__ void transposeKernel(const float *in, float *out, int N)
{
    __shared__ float s_tile[TILE_SIZE][TILE_SIZE + 1]; // padding to avoid bank conflicts

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Load data from global memory to shared memory
    if (x < N && y < N) {
        s_tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    }
    __syncthreads();

    // Write transposed data from shared memory to global memory
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    if (x < N && y < N) {
        out[y * N + x] = s_tile[threadIdx.x][threadIdx.y];
    }
}

int main(int argc, char **argv)
{
    int N = 1024; // default matrix size
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float *h_in  = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    if (!h_in || !h_out) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input matrix with random values
    for (int i = 0; i < N * N; ++i) {
        h_in[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_in  = nullptr;
    float *d_out = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&d_in,  bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_out, bytes));

    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE,
                 (N + TILE_SIZE - 1) / TILE_SIZE);

    // Timing events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Launch kernel
    CHECK_CUDA(cudaEventRecord(start));
    transposeKernel<<<dimGrid, dimBlock>>>(d_in, d_out, N);
    CHECK_CUDA(cudaEventRecord(stop));

    // Wait for kernel to finish
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Verify correctness
    int errors = 0;
    for (int i = 0; i < N && errors < 10; ++i) {
        for (int j = 0; j < N; ++j) {
            if (fabs(h_in[i * N + j] - h_out[j * N + i]) > 1e-5f) {
                ++errors;
                if (errors <= 10) {
                    printf("Mismatch at (%d,%d): host=%f, device=%f\n",
                           i, j, h_in[i * N + j], h_out[j * N + i]);
                }
            }
        }
    }
    if (errors == 0) {
        printf("Matrix transpose verified successfully.\n");
    } else {
        printf("Matrix transpose has %d mismatches.\n", errors);
    }

    // Report timing
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("Transpose kernel time: %f ms\n", ms);

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    free(h_in);
    free(h_out);

    return 0;
}
