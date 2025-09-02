```cuda
/*
Aim of the program: Add a comment explaining what a CUDA stream is and how it relates to overlapping operations.

My thinking:
- The user asked for a CUDA program that includes a comment explaining what a CUDA stream is and how it can be used to overlap operations. 
- To demonstrate this concept, the code will allocate memory, create two CUDA streams, perform an asynchronous memory copy in one stream, launch a kernel in the other stream, and use cudaDeviceSynchronize() to wait for completion.
- The comment explaining CUDA streams will be added inside the code near the stream creation, providing a clear description for anyone reading the source.
- The program will be minimal but functional: a simple vector addition kernel, asynchronous memory copy, and two separate streams to show potential overlap.
- We'll also include basic error checking for robustness.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA errors */
#define CUDA_CHECK(err)                                                      \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(err));                                    \
        exit(EXIT_FAILURE);                                                  \
    }

/* Simple kernel: add two vectors */
__global__ void vecAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}

int main(void)
{
    const int N = 1 << 20;               // 1M elements
    const size_t bytes = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    /* Create two non-blocking streams */
    cudaStream_t streamMemcpy, streamKernel;
    CUDA_CHECK(cudaStreamCreateWithFlags(&streamMemcpy, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&streamKernel, cudaStreamNonBlocking));

    /* Asynchronously copy host to device in streamMemcpy */
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, streamMemcpy));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice, streamMemcpy));

    /* 
     * CUDA Stream explanation:
     * A CUDA stream is a sequence of operations that execute on the device in order. 
     * Operations in different streams may execute concurrently, provided there are no
     * dependencies between them and the hardware resources allow it. By using multiple
     * streams, we can overlap memory transfers (host-to-device or device-to-host)
     * with kernel execution, thereby improving overall performance. 
     * In this example, the memory copy operations are queued in streamMemcpy, while
     * the kernel launch is queued in streamKernel. If the GPU has enough resources,
     * the kernel execution can overlap with the memory copies.
     */

    /* Launch kernel in streamKernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock, 0, streamKernel>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    /* Asynchronously copy result back to host */
    CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost, streamKernel));

    /* Wait for all operations to finish */
    CUDA_CHECK(cudaStreamSynchronize(streamMemcpy));
    CUDA_CHECK(cudaStreamSynchronize(streamKernel));

    /* Verify results (simple check) */
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            fprintf(stderr, "Mismatch at index %d: %f != %f\n", i, h_C[i], h_A[i] + h_B[i]);
            break;
        }
    }
    printf("Vector addition completed successfully.\n");

    /* Clean up */
    CUDA_CHECK(cudaStreamDestroy(streamMemcpy));
    CUDA_CHECK(cudaStreamDestroy(streamKernel));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```