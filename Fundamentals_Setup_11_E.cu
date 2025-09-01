/*
Aim: Add cudaDeviceSynchronize() after the kernel launch and explain in a comment why this is important for timing and error checking.

Thinking: In CUDA, kernel launches are asynchronous with respect to the host thread. This means the host code continues executing immediately after launching the kernel, without waiting for the kernel to finish. When we want to time the kernel execution or check for errors that may occur during kernel execution, we must ensure that the kernel has actually completed before we query the elapsed time or check for errors. 
The function cudaDeviceSynchronize() blocks the host thread until all previously queued device operations (including the kernel) have completed. 
By placing it right after the kernel launch, we guarantee that the kernel has finished executing before we proceed to timing or error checks. 
If we omitted cudaDeviceSynchronize(), the timer could stop before the kernel finishes, giving a misleadingly short runtime. 
Similarly, any errors that occur during kernel execution would only surface on subsequent API calls; cudaDeviceSynchronize() forces an error check at that point, making debugging easier.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main(void)
{
    const int N = 1 << 20;            // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch the kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Add cudaDeviceSynchronize() after kernel launch
    cudaDeviceSynchronize();  // <--- important for timing and error checking

    // Record the stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify a few results
    bool correct = true;
    for (int i = 0; i < 10; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            correct = false;
            break;
        }
    }

    printf("Vector addition %s.\n", correct ? "succeeded" : "failed");
    printf("Elapsed time: %f ms\n", milliseconds);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}