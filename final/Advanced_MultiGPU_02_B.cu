```cuda
/*
Use CUDA streams to launch the work on both GPUs asynchronously from the host's perspective, then wait for both to complete.
--------------------------------------------------------------------
Thinking Process:

1. Determine the number of available GPUs with `cudaGetDeviceCount`.  
2. For the demo, we will use the first two GPUs (if at least two exist).  
3. On each GPU:
   - Set the device with `cudaSetDevice`.
   - Allocate device memory for an input and output array.
   - Create a CUDA stream with `cudaStreamCreate`.
   - Copy input data from host to device asynchronously with the stream.
   - Launch a simple kernel (e.g., element-wise addition) on that stream.
   - Copy the result back to host asynchronously with the same stream.
4. After initiating all work, we use `cudaStreamSynchronize` on each stream to wait for that stream's work to finish.  
   Since each stream lives on a different device, we set the device back to the corresponding GPU before calling `cudaStreamSynchronize`.  
5. Clean up by destroying streams, freeing device memory, and freeing host memory.  
6. Include robust error checking after each CUDA API call and kernel launch to catch issues early.  
7. The program prints a simple confirmation message when both GPUs have finished their tasks.

This design demonstrates asynchronous launch on multiple GPUs using streams and proper synchronization from the host side.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Simple kernel: each thread adds 1 to its element
__global__ void add_one_kernel(int *d_out, const int *d_in, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = d_in[idx] + 1;
    }
}

int main() {
    const int num_gpus_needed = 2;
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count < num_gpus_needed) {
        fprintf(stderr, "Need at least %d GPUs, but only %d available.\n",
                num_gpus_needed, device_count);
        return EXIT_FAILURE;
    }

    const size_t N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(int);

    // Host memory allocation
    int *h_input = (int *)malloc(bytes);
    int *h_output_gpu0 = (int *)malloc(bytes);
    int *h_output_gpu1 = (int *)malloc(bytes);
    if (!h_input || !h_output_gpu0 || !h_output_gpu1) {
        fprintf(stderr, "Host memory allocation failed.\n");
        return EXIT_FAILURE;
    }

    // Initialize input
    for (size_t i = 0; i < N; ++i) {
        h_input[i] = (int)i;
    }

    // Device 0
    int dev0 = 0;
    CHECK_CUDA(cudaSetDevice(dev0));
    int *d_in0, *d_out0;
    CHECK_CUDA(cudaMalloc(&d_in0, bytes));
    CHECK_CUDA(cudaMalloc(&d_out0, bytes));

    cudaStream_t stream0;
    CHECK_CUDA(cudaStreamCreate(&stream0));

    // Device 1
    int dev1 = 1;
    CHECK_CUDA(cudaSetDevice(dev1));
    int *d_in1, *d_out1;
    CHECK_CUDA(cudaMalloc(&d_in1, bytes));
    CHECK_CUDA(cudaMalloc(&d_out1, bytes));

    cudaStream_t stream1;
    CHECK_CUDA(cudaStreamCreate(&stream1));

    // Copy input to device 0 asynchronously
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaMemcpyAsync(d_in0, h_input, bytes, cudaMemcpyHostToDevice, stream0));

    // Copy input to device 1 asynchronously
    CHECK_CUDA(cudaSetDevice(dev1));
    CHECK_CUDA(cudaMemcpyAsync(d_in1, h_input, bytes, cudaMemcpyHostToDevice, stream1));

    // Launch kernels on both devices asynchronously
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    CHECK_CUDA(cudaSetDevice(dev0));
    add_one_kernel<<<gridSize, blockSize, 0, stream0>>>(d_out0, d_in0, N);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaSetDevice(dev1));
    add_one_kernel<<<gridSize, blockSize, 0, stream1>>>(d_out1, d_in1, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy results back asynchronously
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaMemcpyAsync(h_output_gpu0, d_out0, bytes, cudaMemcpyDeviceToHost, stream0));

    CHECK_CUDA(cudaSetDevice(dev1));
    CHECK_CUDA(cudaMemcpyAsync(h_output_gpu1, d_out1, bytes, cudaMemcpyDeviceToHost, stream1));

    // Synchronize streams (wait for completion)
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaStreamSynchronize(stream0));

    CHECK_CUDA(cudaSetDevice(dev1));
    CHECK_CUDA(cudaStreamSynchronize(stream1));

    // Verify results (simple check)
    bool success = true;
    for (size_t i = 0; i < N; ++i) {
        if (h_output_gpu0[i] != h_input[i] + 1 || h_output_gpu1[i] != h_input[i] + 1) {
            success = false;
            printf("Mismatch at index %zu: device0=%d, device1=%d, expected=%d\n",
                   i, h_output_gpu0[i], h_output_gpu1[i], h_input[i] + 1);
            break;
        }
    }
    if (success) {
        printf("Success: Both GPUs completed their work correctly.\n");
    } else {
        printf("Failure: Mismatch found.\n");
    }

    // Cleanup
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaStreamDestroy(stream0));
    CHECK_CUDA(cudaFree(d_in0));
    CHECK_CUDA(cudaFree(d_out0));

    CHECK_CUDA(cudaSetDevice(dev1));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaFree(d_in1));
    CHECK_CUDA(cudaFree(d_out1));

    free(h_input);
    free(h_output_gpu0);
    free(h_output_gpu1);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
```