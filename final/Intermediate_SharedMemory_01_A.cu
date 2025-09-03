/*
In the naive kernel, analyze the memory access pattern for matrix B. Explain why it is inefficient (uncoalesced/strided access).

The naive matrix multiplication kernel assigns one thread per output element C[i][j]. Each thread
iterates over the shared dimension k to accumulate the dot product of row i of A and column j of B.
The address of B used in this inner loop is typically computed as B[k * N + j] (assuming row‑major
layout for both matrices). For a fixed j, the index k * N + j increases by N for each increment of
k. Thus, when a warp of 32 threads simultaneously accesses B with different values of j, each
thread in the warp reads from memory locations that are N apart. This is a *strided* access pattern
with a stride equal to the number of columns in B.

GPUs are highly efficient when consecutive threads in a warp read consecutive memory words so
that the hardware can coalesce these accesses into a single transaction. In contrast, the strided
pattern forces the hardware to issue many separate memory transactions because the addresses are
not contiguous. This leads to increased memory traffic, higher latency, and poor bandwidth
utilization—i.e., the accesses to matrix B are uncoalesced and inefficient.

To see this in practice, we implement the naive kernel below. By inspecting the generated
PTX (or by running with profiling tools), one can confirm the uncoalesced accesses to B. The
comment in the kernel explains the pattern and its inefficiency. In a production kernel one
would instead use a tiled approach that loads blocks of B into shared memory so that each warp
accesses consecutive elements, thereby achieving coalesced memory traffic.

The program below performs a simple matrix multiplication using the naive approach, prints a
brief explanation of the memory access pattern, and verifies correctness on a small test case.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Matrix dimensions (small for demonstration)
#define M 4   // rows of A and C
#define N 4   // columns of B and rows of B

// Naive matrix multiplication kernel
__global__ void matMulNaive(const float *A, const float *B, float *C, int m, int n) {
    // Each thread computes one element of C
    int row = blockIdx.y * blockDim.y + threadIdx.y; // i
    int col = blockIdx.x * blockDim.x + threadIdx.x; // j

    if (row < m && col < n) {
        float sum = 0.0f;
        // Compute dot product of row i of A and column j of B
        for (int k = 0; k < n; ++k) {
            // Access pattern for B: B[k * n + col]
            // For a fixed 'col', the index increases by 'n' each iteration,
            // i.e., stride = n. When a warp accesses different 'col' values,
            // each thread reads from memory locations that are separated by 'n'.
            // This strided pattern leads to uncoalesced memory accesses.
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Host function to initialize matrices
void initMatrix(float *mat, int rows, int cols, float value = 1.0f) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = value;
    }
}

// Simple check for CUDA errors
void checkCudaError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Allocate host matrices
    size_t size = M * N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_C_ref = (float *)malloc(size);

    // Initialize matrices
    initMatrix(h_A, M, N, 1.0f);          // all ones
    initMatrix(h_B, M, N, 2.0f);          // all twos
    initMatrix(h_C, M, N, 0.0f);          // zero
    initMatrix(h_C_ref, M, N, 0.0f);      // zero

    // Compute reference result on host
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += h_A[i * N + k] * h_B[k * N + j];
            }
            h_C_ref[i * N + j] = sum;
        }
    }

    // Allocate device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    checkCudaError("cudaMalloc");

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    checkCudaError("cudaMemcpy H2D");

    // Define block and grid sizes
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    matMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N);
    checkCudaError("Kernel launch");

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy D2H");

    // Verify correctness
    bool correct = true;
    for (int i = 0; i < M * N; ++i) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("Result is correct.\n");
    } else {
        printf("Result is incorrect.\n");
    }

    // Free resources
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
