/*
Aim: Profile this corrected naive transpose kernel.

Thinking:
- The task is to measure the execution time of a simple, correct naive transpose kernel written in CUDA C.
- The naive transpose will copy each element from the input matrix to its transposed position in the output matrix. It does not use shared memory or any other optimizations.
- We'll set up a square matrix of a configurable size, initialize it on the host, copy it to the device, run the transpose kernel, and copy the result back.
- We'll use CUDA events to time only the kernel execution, excluding memory transfers.
- We'll also perform a simple correctness check by transposing the matrix on the host and comparing the results.
- The code will include a macro for CUDA error checking to aid debugging.
- The block dimensions will be 16x16 threads for generality.
- The program will print the measured time in milliseconds and report whether the transpose was correct.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",           \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Naive transpose kernel
__global__ void transposeNaive(const float *in, float *out, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (x < width && y < height) {
        out[y + x * height] = in[x + y * width];
    }
}

// Host function to perform transpose for correctness check
void cpuTranspose(const float *in, float *out, int width, int height)
{
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            out[y + x * height] = in[x + y * width];
        }
    }
}

int main(int argc, char *argv[])
{
    // Matrix dimensions (square for simplicity)
    int N = 1024; // default size
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            fprintf(stderr, "Invalid matrix size.\n");
            return EXIT_FAILURE;
        }
    }

    size_t size = N * N * sizeof(float);
    printf("Matrix size: %d x %d\n", N, N);
    printf("Allocating %zu MB of host memory.\n", size / (1024 * 1024));

    // Allocate host memory
    float *h_in  = (float *)malloc(size);
    float *h_out = (float *)malloc(size);
    float *h_ref = (float *)malloc(size); // for CPU reference

    if (!h_in || !h_out || !h_ref) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input matrix with some pattern
    for (int i = 0; i < N * N; ++i) {
        h_in[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_in  = nullptr;
    float *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_in,  size));
    CUDA_CHECK(cudaMalloc((void **)&d_out, size));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);

    // Timing with CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch kernel
    transposeNaive<<<gridSize, blockSize>>>(d_in, d_out, N, N);

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));

    // Wait for completion
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Compute elapsed time
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("Kernel execution time: %f ms\n", ms);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // Compute reference transpose on CPU
    cpuTranspose(h_in, h_ref, N, N);

    // Verify correctness
    bool correct = true;
    for (int i = 0; i < N * N; ++i) {
        if (fabs(h_out[i] - h_ref[i]) > 1e-5f) {
            correct = false;
            printf("Mismatch at index %d: GPU=%f CPU=%f\n", i, h_out[i], h_ref[i]);
            break;
        }
    }
    printf("Transpose correctness: %s\n", correct ? "PASSED" : "FAILED");

    // Clean up
    free(h_in);
    free(h_out);
    free(h_ref);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
