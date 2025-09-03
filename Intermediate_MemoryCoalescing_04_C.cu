/*
Now test a strided write: `output[i * STRIDE] = input[i]`. Profile and compare.

The goal of this program is to measure and compare the performance of two simple CUDA kernels:
1. A regular copy kernel that writes directly to consecutive memory locations.
2. A strided write kernel that writes each element to a memory location offset by a fixed stride.

To do this, the program:
- Allocates a large input array on the host, fills it with sequential values.
- Allocates two output arrays on the device: one for the regular copy and one for the strided write. The strided output array is sized `N * STRIDE` to avoid out‑of‑bounds writes.
- Launches both kernels with the same grid/block configuration and times each using CUDA events.
- Copies the results back to the host and verifies correctness.
- Prints the elapsed time for each kernel and whether the outputs are correct.

The program uses CUDA error checking helpers to catch and report any runtime errors. It also synchronizes the device after each kernel launch to ensure accurate timing. The results can be used to see how memory access patterns (contiguous vs strided) affect performance on the GPU.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N (1 << 20)          // Number of elements in input array
#define STRIDE 4             // Stride factor for the strided write
#define BLOCK_SIZE 256       // Threads per block

// Error checking macro
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Kernel that writes each input element to the output array at index i
__global__ void copyKernel(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

// Kernel that writes each input element to the output array at index i * STRIDE
__global__ void stridedCopyKernel(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx * STRIDE] = input[idx];
    }
}

// Host function to compare two arrays; returns 1 if equal, 0 otherwise
int compareArrays(const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            return 0;
        }
    }
    return 1;
}

int main(void) {
    size_t inputSize = N * sizeof(float);
    size_t outputSize = N * sizeof(float);
    size_t stridedOutputSize = N * STRIDE * sizeof(float);

    // Allocate host memory
    float *h_input = (float*)malloc(inputSize);
    float *h_output = (float*)malloc(outputSize);
    float *h_stridedOutput = (float*)malloc(stridedOutputSize);

    if (!h_input || !h_output || !h_stridedOutput) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input array
    for (size_t i = 0; i < N; ++i) {
        h_input[i] = (float)i;
    }

    // Allocate device memory
    float *d_input = NULL;
    float *d_output = NULL;
    float *d_stridedOutput = NULL;

    CUDA_CHECK(cudaMalloc((void**)&d_input, inputSize));
    CUDA_CHECK(cudaMalloc((void**)&d_output, outputSize));
    CUDA_CHECK(cudaMalloc((void**)&d_stridedOutput, stridedOutputSize));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice));

    // Define grid dimensions
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Create CUDA events for timing
    cudaEvent_t startCopy, stopCopy;
    cudaEvent_t startStrided, stopStrided;
    CUDA_CHECK(cudaEventCreate(&startCopy));
    CUDA_CHECK(cudaEventCreate(&stopCopy));
    CUDA_CHECK(cudaEventCreate(&startStrided));
    CUDA_CHECK(cudaEventCreate(&stopStrided));

    // Launch copy kernel
    CUDA_CHECK(cudaEventRecord(startCopy, 0));
    copyKernel<<<gridDim, blockDim>>>(d_input, d_output, N);
    CUDA_CHECK(cudaEventRecord(stopCopy, 0));

    // Launch strided copy kernel
    CUDA_CHECK(cudaEventRecord(startStrided, 0));
    stridedCopyKernel<<<gridDim, blockDim>>>(d_input, d_stridedOutput, N);
    CUDA_CHECK(cudaEventRecord(stopStrided, 0));

    // Wait for kernels to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_stridedOutput, d_stridedOutput, stridedOutputSize, cudaMemcpyDeviceToHost));

    // Verify results
    int correctCopy = compareArrays(h_input, h_output, N);
    int correctStrided = 1;
    for (size_t i = 0; i < N; ++i) {
        if (h_stridedOutput[i * STRIDE] != h_input[i]) {
            correctStrided = 0;
            break;
        }
    }

    // Calculate elapsed time
    float msCopy, msStrided;
    CUDA_CHECK(cudaEventElapsedTime(&msCopy, startCopy, stopCopy));
    CUDA_CHECK(cudaEventElapsedTime(&msStrided, startStrided, stopStrided));

    // Print results
    printf("Regular copy kernel time: %f ms\n", msCopy);
    printf("Strided copy kernel time: %f ms\n", msStrided);
    printf("Regular copy correct: %s\n", correctCopy ? "YES" : "NO");
    printf("Strided copy correct: %s\n", correctStrided ? "YES" : "NO");

    // Clean up
    CUDA_CHECK(cudaEventDestroy(startCopy));
    CUDA_CHECK(cudaEventDestroy(stopCopy));
    CUDA_CHECK(cudaEventDestroy(startStrided));
    CUDA_CHECK(cudaEventDestroy(stopStrided));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_stridedOutput));

    free(h_input);
    free(h_output);
    free(h_stridedOutput);

    return 0;
}
