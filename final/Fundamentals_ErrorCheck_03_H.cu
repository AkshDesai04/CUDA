/*
In a program with multiple streams, `cudaDeviceSynchronize` is a heavyweight operation. What is a more fine-grained alternative? (`cudaStreamSynchronize`).

Thinking:
The goal of this example is to illustrate how to use multiple CUDA streams and to show that instead of waiting for the entire device to finish all work with cudaDeviceSynchronize, we can wait only on the specific streams that we need with cudaStreamSynchronize. The code will:
1. Allocate a large array on the device.
2. Create two CUDA streams.
3. Launch two kernels that write to different halves of the array, each in its own stream.
4. Synchronize each stream individually using cudaStreamSynchronize.
5. Copy the result back to the host and print a few elements to verify correctness.
6. Include error checking and a macro to simplify CUDA API error handling.

The example will compile as a single .cu file and demonstrates the fine-grained synchronization approach described in the prompt. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N (1<<20)           // 1M elements
#define BLOCK_SIZE 256

// Simple error checking macro
#define CUDA_CHECK(call)                                               \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Kernel that fills part of an array with a value based on stream ID
__global__ void fill_kernel(int *data, int start, int end, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < end) {
        data[idx] = value;
    }
}

int main(void) {
    int *d_array;
    int *h_array = (int*)malloc(N * sizeof(int));
    if (!h_array) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_array, N * sizeof(int)));

    // Create two streams
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // Launch kernel in stream1 to fill first half with 1
    int mid = N / 2;
    int threads = BLOCK_SIZE;
    int blocks = (mid + threads - 1) / threads;
    fill_kernel<<<blocks, threads, 0, stream1>>>(d_array, 0, mid, 1);
    CUDA_CHECK(cudaGetLastError());

    // Launch kernel in stream2 to fill second half with 2
    blocks = ((N - mid) + threads - 1) / threads;
    fill_kernel<<<blocks, threads, 0, stream2>>>(d_array, mid, N, 2);
    CUDA_CHECK(cudaGetLastError());

    // Fine‑grained synchronization: wait only for each stream
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    // Alternatively, the heavyweight approach would be:
    // CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify a few values
    printf("First 10 elements: ");
    for (int i = 0; i < 10; ++i) printf("%d ", h_array[i]);
    printf("\n");

    printf("Last 10 elements: ");
    for (int i = N-10; i < N; ++i) printf("%d ", h_array[i]);
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaFree(d_array));
    free(h_array);

    return 0;
}
