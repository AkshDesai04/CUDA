/*
Use `cudaMallocPitch` and `cudaMemcpy2D` for your matrices and modify the kernel access logic to use the pitch.
I wanted to demonstrate how to allocate 2‑D device memory with an optimal pitch (row stride) and transfer data to and from that memory using cudaMemcpy2D.  The kernel must be adapted to use the pitch in bytes to index the elements correctly.  For clarity I chose a simple element‑wise addition of two float matrices A and B producing matrix C.  The code:
- Defines matrix dimensions.
- Allocates and initializes host arrays.
- Uses cudaMallocPitch to allocate pitched device memory for A, B, and C.
- Uses cudaMemcpy2D to copy host matrices to device and back.
- Launches a kernel that computes C[row][col] = A[row][col] + B[row][col] using the pitch.
- Frees device memory.
- Prints a few elements of the result for verification.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

__global__ void addKernel(const float *A, const float *B, float *C,
                          size_t pitch, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // y-coordinate
    int col = blockIdx.x * blockDim.x + threadIdx.x; // x-coordinate

    if (row < height && col < width) {
        // Convert pitch from bytes to number of float elements
        int pitchElements = pitch / sizeof(float);

        // Compute linear offset for pitched memory
        float a = A[row * pitchElements + col];
        float b = B[row * pitchElements + col];
        C[row * pitchElements + col] = a + b;
    }
}

int main(void) {
    const int width  = 512;   // number of columns
    const int height = 512;   // number of rows

    size_t matrixSize = width * height * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(matrixSize);
    float *h_B = (float*)malloc(matrixSize);
    float *h_C = (float*)malloc(matrixSize);

    // Initialize host matrices
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * width + j;
            h_A[idx] = (float)(i + j);
            h_B[idx] = (float)(i - j);
        }
    }

    // Device pointers
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    size_t pitchA, pitchB, pitchC;

    // Allocate pitched device memory
    CHECK_CUDA(cudaMallocPitch(&d_A, &pitchA, width * sizeof(float), height));
    CHECK_CUDA(cudaMallocPitch(&d_B, &pitchB, width * sizeof(float), height));
    CHECK_CUDA(cudaMallocPitch(&d_C, &pitchC, width * sizeof(float), height));

    // Copy data from host to device using 2D copy
    CHECK_CUDA(cudaMemcpy2D(d_A, pitchA,
                            h_A, width * sizeof(float),
                            width * sizeof(float), height,
                            cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy2D(d_B, pitchB,
                            h_B, width * sizeof(float),
                            width * sizeof(float), height,
                            cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    addKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C,
                                             pitchC, width, height);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy2D(h_C, width * sizeof(float),
                            d_C, pitchC,
                            width * sizeof(float), height,
                            cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("Sample results:\n");
    for (int i = 0; i < 5; ++i) {
        printf("C[%d,%d] = %f\n", i, i, h_C[i * width + i]);
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
