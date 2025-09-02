/*
Aim: test a strided write: `output[i * STRIDE] = input[i]`. Profile and compare.

Thinking:
- Implement two CUDA kernels: one performs a contiguous copy (`output[i] = input[i]`), the other performs a strided write (`output[i * STRIDE] = input[i]`).
- Use identical input data for both kernels to ensure a fair comparison.
- Allocate device buffers: `d_input`, `d_output_contiguous`, and `d_output_strided`. The strided output buffer is sized `N * STRIDE` to hold the maximum index `i * STRIDE`.
- Launch each kernel with the same number of threads (`threadsPerBlock = 256`) and blocks (`gridDim = (N + threadsPerBlock - 1) / threadsPerBlock`).
- Measure execution time of each kernel using CUDA events (`cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`).
- After kernel execution, copy the device results back to host memory and validate correctness by comparing each element of the output arrays with the original input data.
- Report the elapsed times and compute simple throughput metrics (bytes processed per second) for both kernels.
- The program is self‑contained and can be compiled with `nvcc test_strided_write.cu -o test_strided_write`.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                      \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                              \
        }                                                                     \
    } while (0)

// Size of the input array
#define N (1 << 20)     // 1,048,576 elements
// Stride for the strided kernel
#define STRIDE 4

__global__ void kernelContiguous(const float* input, float* output, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

__global__ void kernelStrided(const float* input, float* output, size_t n, int stride)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx * stride] = input[idx];
    }
}

int main(void)
{
    size_t size = N * sizeof(float);
    size_t stridedSize = N * STRIDE * sizeof(float);

    // Allocate host memory
    float *h_input = (float*)malloc(size);
    float *h_output_contiguous = (float*)malloc(size);
    float *h_output_strided = (float*)malloc(stridedSize);

    if (!h_input || !h_output_contiguous || !h_output_strided) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input data
    for (size_t i = 0; i < N; ++i) {
        h_input[i] = (float)i;
    }

    // Allocate device memory
    float *d_input = NULL;
    float *d_output_contiguous = NULL;
    float *d_output_strided = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_input, size));
    CHECK_CUDA(cudaMalloc((void**)&d_output_contiguous, size));
    CHECK_CUDA(cudaMalloc((void**)&d_output_strided, stridedSize));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Define kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Events for timing
    cudaEvent_t startCont, stopCont, startStr, stopStr;
    CHECK_CUDA(cudaEventCreate(&startCont));
    CHECK_CUDA(cudaEventCreate(&stopCont));
    CHECK_CUDA(cudaEventCreate(&startStr));
    CHECK_CUDA(cudaEventCreate(&stopStr));

    // Launch contiguous kernel
    CHECK_CUDA(cudaEventRecord(startCont, 0));
    kernelContiguous<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output_contiguous, N);
    CHECK_CUDA(cudaEventRecord(stopCont, 0));

    // Launch strided kernel
    CHECK_CUDA(cudaEventRecord(startStr, 0));
    kernelStrided<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output_strided, N, STRIDE);
    CHECK_CUDA(cudaEventRecord(stopStr, 0));

    // Wait for kernels to finish
    CHECK_CUDA(cudaEventSynchronize(stopCont));
    CHECK_CUDA(cudaEventSynchronize(stopStr));

    // Calculate elapsed times
    float timeCont = 0.0f, timeStr = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&timeCont, startCont, stopCont));
    CHECK_CUDA(cudaEventElapsedTime(&timeStr, startStr, stopStr));

    // Copy outputs back to host
    CHECK_CUDA(cudaMemcpy(h_output_contiguous, d_output_contiguous, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_strided, d_output_strided, stridedSize, cudaMemcpyDeviceToHost));

    // Verify results
    int errorsCont = 0;
    for (size_t i = 0; i < N; ++i) {
        if (h_output_contiguous[i] != h_input[i]) {
            errorsCont++;
            if (errorsCont <= 10) {
                fprintf(stderr, "Contiguous mismatch at index %zu: host=%f, device=%f\n",
                        i, h_input[i], h_output_contiguous[i]);
            }
        }
    }

    int errorsStr = 0;
    for (size_t i = 0; i < N; ++i) {
        size_t outIdx = i * STRIDE;
        if (outIdx >= N * STRIDE) break;
        if (h_output_strided[outIdx] != h_input[i]) {
            errorsStr++;
            if (errorsStr <= 10) {
                fprintf(stderr, "Strided mismatch at index %zu (output index %zu): host=%f, device=%f\n",
                        i, outIdx, h_input[i], h_output_strided[outIdx]);
            }
        }
    }

    // Report
    printf("Contiguous kernel time:   %.3f ms\n", timeCont);
    printf("Strided   kernel time:   %.3f ms\n", timeStr);
    printf("Contiguous errors: %d\n", errorsCont);
    printf("Strided errors: %d\n", errorsStr);

    // Compute simple bandwidth (bytes transferred)
    double bytesCont = 2.0 * N * sizeof(float);   // read + write
    double bytesStr = 2.0 * N * sizeof(float);    // same amount of data processed
    printf("Contiguous bandwidth: %.3f GB/s\n", bytesCont / timeCont / 1e6);
    printf("Strided   bandwidth: %.3f GB/s\n", bytesStr / timeStr / 1e6);

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(startCont));
    CHECK_CUDA(cudaEventDestroy(stopCont));
    CHECK_CUDA(cudaEventDestroy(startStr));
    CHECK_CUDA(cudaEventDestroy(stopStr));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output_contiguous));
    CHECK_CUDA(cudaFree(d_output_strided));
    free(h_input);
    free(h_output_contiguous);
    free(h_output_strided);

    return EXIT_SUCCESS;
}
