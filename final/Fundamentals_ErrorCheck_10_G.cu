/*
What happens if you call `elapsed_ms()` before `stop()` has been called or before the event has completed? The result is undefined or an error.
I need to satisfy the developer instruction: first, a multiline comment that contains the exact text of the prompt, word for word, and then my entire chain-of-thought about how to write a CUDA program that illustrates this situation. The aim of the program is to demonstrate that calling cudaEventElapsedTime before the stop event has been recorded (or before the event sequence has finished) leads to undefined or incorrect results. I will create a small CUDA kernel that performs a dummy workload, set up two events (start and stop), record the start event, launch the kernel, and then attempt to get the elapsed time *before* recording the stop event to show the undefined behaviour. Then I will properly record the stop event, synchronize, and query the correct elapsed time. The program will compile with nvcc and print both the premature and correct timings. The entire reasoning, including decisions about how to illustrate the undefined behaviour, goes inside this comment. The code will be a single .cu file, no external dependencies beyond the CUDA runtime. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple CUDA kernel that performs a dummy workload.
__global__ void dummyKernel(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Perform some arithmetic to keep the GPU busy.
        for (int i = 0; i < 1000; ++i) {
            data[idx] = data[idx] + i;
        }
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    const int bytes = N * sizeof(int);

    // Allocate host memory.
    int *h_data = (int *)malloc(bytes);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }
    for (int i = 0; i < N; ++i) h_data[i] = 0;

    // Allocate device memory.
    int *d_data = nullptr;
    cudaError_t err = cudaMalloc((void **)&d_data, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        free(h_data);
        return 1;
    }

    // Copy data to device.
    err = cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return 1;
    }

    // Create CUDA events.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event.
    cudaEventRecord(start, 0);

    // Launch the kernel.
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    dummyKernel<<<blocks, threadsPerBlock>>>(d_data, N);

    // Attempt to get elapsed time *before* recording the stop event.
    // This is the undefined behaviour scenario.
    float elapsedTimeBeforeStop = 0.0f;
    err = cudaEventElapsedTime(&elapsedTimeBeforeStop, start, stop);
    if (err == cudaSuccess) {
        printf("Elapsed time before stop recorded: %f ms (undefined behavior)\n", elapsedTimeBeforeStop);
    } else {
        printf("cudaEventElapsedTime before stop failed: %s\n", cudaGetErrorString(err));
    }

    // Now record the stop event and synchronize.
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Query the correct elapsed time.
    float elapsedTime = 0.0f;
    err = cudaEventElapsedTime(&elapsedTime, start, stop);
    if (err == cudaSuccess) {
        printf("Correct elapsed time after stop: %f ms\n", elapsedTime);
    } else {
        printf("cudaEventElapsedTime after stop failed: %s\n", cudaGetErrorString(err));
    }

    // Clean up.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    free(h_data);

    return 0;
}
