/*
Use CUDA streams to launch the work on both GPUs asynchronously from the host's perspective, then wait for both to complete.

My approach:
1. Query the number of GPUs; exit with an error if fewer than 2.
2. Allocate a simple host array and initialize it.
3. For each of the first two GPUs:
   a. Set the current device.
   b. Allocate device memory for that GPU.
   c. Create a stream on that device.
   d. Asynchronously copy the host array to device memory using the stream.
   e. Launch a simple kernel that adds 1.0f to each element, also on that stream.
   f. Asynchronously copy the results back to a separate host array using the same stream.
4. After launching all operations, synchronously wait on each stream to ensure completion.
5. Verify that the results on both GPUs are correct.
6. Clean up all allocated memory and streams.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// Simple kernel that adds 1.0f to each element
__global__ void addOne(float* d_data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_data[idx] += 1.0f;
    }
}

int main() {
    const int N = 1 << 20;           // 1M elements
    const int THREADS_PER_BLOCK = 256;

    // Allocate and initialize host input array
    float* h_input = (float*)malloc(N * sizeof(float));
    if (!h_input) {
        fprintf(stderr, "Failed to allocate host input array.\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // Allocate host output arrays for each GPU
    float* h_output[2];
    for (int i = 0; i < 2; ++i) {
        h_output[i] = (float*)malloc(N * sizeof(float));
        if (!h_output[i]) {
            fprintf(stderr, "Failed to allocate host output array for GPU %d.\n", i);
            return EXIT_FAILURE;
        }
    }

    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        fprintf(stderr, "This program requires at least 2 GPUs. Found %d.\n", deviceCount);
        return EXIT_FAILURE;
    }

    // Device pointers and streams
    float* d_input[2];
    cudaStream_t stream[2];

    for (int dev = 0; dev < 2; ++dev) {
        CHECK_CUDA(cudaSetDevice(dev));

        // Allocate device memory
        CHECK_CUDA(cudaMalloc((void**)&d_input[dev], N * sizeof(float)));

        // Create stream
        CHECK_CUDA(cudaStreamCreate(&stream[dev]));

        // Asynchronously copy input from host to device
        CHECK_CUDA(cudaMemcpyAsync(d_input[dev], h_input, N * sizeof(float),
                                   cudaMemcpyHostToDevice, stream[dev]));

        // Launch kernel on the stream
        int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        addOne<<<blocks, THREADS_PER_BLOCK, 0, stream[dev]>>>(d_input[dev], N);
        // Check for kernel launch errors
        CHECK_CUDA(cudaGetLastError());

        // Asynchronously copy result back to host
        CHECK_CUDA(cudaMemcpyAsync(h_output[dev], d_input[dev], N * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream[dev]));
    }

    // Wait for all streams to complete
    for (int dev = 0; dev < 2; ++dev) {
        CHECK_CUDA(cudaSetDevice(dev));
        CHECK_CUDA(cudaStreamSynchronize(stream[dev]));
        CHECK_CUDA(cudaStreamDestroy(stream[dev]));
        CHECK_CUDA(cudaFree(d_input[dev]));
    }

    // Verify results
    for (int dev = 0; dev < 2; ++dev) {
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float expected = h_input[i] + 1.0f;
            if (fabs(h_output[dev][i] - expected) > 1e-5f) {
                fprintf(stderr, "Verification failed on GPU %d at index %d: got %f, expected %f\n",
                        dev, i, h_output[dev][i], expected);
                ok = false;
                break;
            }
        }
        if (ok) {
            printf("GPU %d: Verification succeeded.\n", dev);
        } else {
            printf("GPU %d: Verification failed.\n", dev);
        }
    }

    // Clean up host memory
    free(h_input);
    for (int i = 0; i < 2; ++i) {
        free(h_output[i]);
    }

    return EXIT_SUCCESS;
}
