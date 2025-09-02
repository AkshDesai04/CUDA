```cuda
/*
Aim: Write a version of the function for streams: `checkStream(cudaStream_t stream)` which calls `cudaStreamSynchronize`.

Thinking:
To create a minimal, self‑contained CUDA program that demonstrates how to synchronize a user‑defined stream, we need:
1. Include `<cuda_runtime.h>` for CUDA runtime API definitions.
2. Implement `checkStream` which:
   - Calls `cudaStreamSynchronize(stream)`.
   - Checks the returned `cudaError_t`.
   - Prints an error message (via `printf`) if the synchronization fails.
   - Returns the error code so the caller can react appropriately.
3. Provide a small `main` routine that:
   - Creates a stream with `cudaStreamCreate`.
   - Launches a trivial kernel (e.g., adding two arrays) to generate work in that stream.
   - Calls `checkStream` to synchronize the stream and confirm all work finished.
   - Destroys the stream.
   - Reports overall success or failure.
4. Keep the code straightforward so it can be compiled with `nvcc` and run on any device with CUDA support.

The resulting file is a complete `.cu` source that compiles and demonstrates the requested functionality.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Simple CUDA kernel that does nothing but can be used to generate work in a stream
__global__ void dummyKernel(int *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + 1;  // trivial operation
    }
}

/**
 * @brief Synchronize a CUDA stream and report any error.
 *
 * @param stream The CUDA stream to synchronize.
 * @return cudaError_t CUDA error code returned by cudaStreamSynchronize.
 */
cudaError_t checkStream(cudaStream_t stream)
{
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        printf("Error synchronizing stream: %s\n", cudaGetErrorString(err));
    }
    return err;
}

int main()
{
    const int N = 1 << 20; // 1M elements
    int *d_data = nullptr;

    // Allocate device memory
    if (cudaMalloc((void**)&d_data, N * sizeof(int))) {
        printf("cudaMalloc failed\n");
        return -1;
    }

    // Create a stream
    cudaStream_t stream;
    if (cudaStreamCreate(&stream)) {
        printf("cudaStreamCreate failed\n");
        cudaFree(d_data);
        return -1;
    }

    // Launch kernel in the created stream
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    dummyKernel<<<blocks, threadsPerBlock, 0, stream>>>(d_data, N);

    // Check for kernel launch errors
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(launchErr));
        cudaStreamDestroy(stream);
        cudaFree(d_data);
        return -1;
    }

    // Synchronize the stream using our helper
    if (checkStream(stream) != cudaSuccess) {
        printf("Stream synchronization failed\n");
        cudaStreamDestroy(stream);
        cudaFree(d_data);
        return -1;
    }

    // Clean up
    cudaStreamDestroy(stream);
    cudaFree(d_data);

    printf("All operations completed successfully.\n");
    return 0;
}
```