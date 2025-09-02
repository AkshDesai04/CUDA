```cpp
/*
Use Nsight Systems (`nsys`) to visualize the timeline. You should now see the HtoD copy for chunk `k+1` overlapping with the kernel execution for chunk `k`.

The goal of this program is to demonstrate how to overlap host‑to‑device (HtoD) memory copies with kernel execution using CUDA streams. By running this code under Nsight Systems (e.g., `nsys profile --stats=timeline ./pipeline`), you should observe that the copy operation for the next chunk begins while the kernel for the current chunk is still running. 

To achieve this, the program:
1. Allocates pinned host memory for fast asynchronous copies.
2. Creates two device buffers (double buffering) and two CUDA streams: one dedicated to data transfer (streamCopy) and one to kernel execution (streamKernel).
3. In a loop over data chunks, it:
   - Initiates an async copy of the next chunk into the device buffer that is not currently being processed.
   - Launches the kernel on the device buffer that was just filled.
   - The copy of the next chunk can start while the kernel on the current chunk is executing, providing overlap.
4. After processing all chunks, it synchronizes and copies the results back to host.

The kernel simply increments each element by one. This is not the focus; the focus is the pipeline and overlapping of operations.

Compile with: `nvcc -O2 -o pipeline pipeline.cu`
Run under Nsight Systems: `nsys profile --stats=timeline ./pipeline`

The timeline will show two streams and overlapping events: 
- A HtoD copy for chunk `k+1` on streamCopy.
- A kernel launch for chunk `k` on streamKernel.
Both events will run concurrently, demonstrating effective pipelining.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Simple kernel: increment each element by 1
__global__ void add_one(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

int main() {
    const int N = 1 << 24;          // Total number of elements (~16M)
    const int numChunks = 8;        // Number of chunks to pipeline
    const int chunkSize = N / numChunks;
    const int remainder = N % numChunks; // Handle any leftover elements

    // Allocate pinned host memory for input and output
    float *h_input = nullptr;
    float *h_output = nullptr;
    CUDA_CHECK(cudaHostAlloc((void**)&h_input,  N * sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_output, N * sizeof(float), cudaHostAllocDefault));

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // Device buffers (double buffering)
    float *d_buf[2];
    CUDA_CHECK(cudaMalloc(&d_buf[0], chunkSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_buf[1], chunkSize * sizeof(float)));

    // Create streams: one for copy, one for kernel
    cudaStream_t streamCopy, streamKernel;
    CUDA_CHECK(cudaStreamCreate(&streamCopy));
    CUDA_CHECK(cudaStreamCreate(&streamKernel));

    int currentBuf = 0; // Index of buffer being processed

    // Start by copying the first chunk
    CUDA_CHECK(cudaMemcpyAsync(d_buf[0], h_input, chunkSize * sizeof(float),
                               cudaMemcpyHostToDevice, streamCopy));

    // Process each chunk
    for (int chunk = 0; chunk < numChunks; ++chunk) {
        // Determine size for this chunk (handle last chunk if remainder > 0)
        int thisChunkSize = chunkSize;
        if (chunk == numChunks - 1) {
            thisChunkSize += remainder;
        }

        // Wait for the copy of the current chunk to finish before launching kernel
        CUDA_CHECK(cudaStreamSynchronize(streamCopy));

        // Launch kernel on the buffer that just received data
        int threadsPerBlock = 256;
        int blocksPerGrid = (thisChunkSize + threadsPerBlock - 1) / threadsPerBlock;
        add_one<<<blocksPerGrid, threadsPerBlock, 0, streamKernel>>>(d_buf[currentBuf], thisChunkSize);
        CUDA_CHECK(cudaGetLastError());

        // If there is a next chunk, start copying it into the alternate buffer
        if (chunk + 1 < numChunks) {
            int nextBuf = 1 - currentBuf;
            int nextChunkSize = chunkSize;
            if (chunk + 1 == numChunks - 1) {
                nextChunkSize += remainder;
            }
            const float *src = h_input + (chunk + 1) * chunkSize;
            CUDA_CHECK(cudaMemcpyAsync(d_buf[nextBuf], src,
                                       nextChunkSize * sizeof(float),
                                       cudaMemcpyHostToDevice, streamCopy));
        }

        // Copy result back to host from the buffer after kernel completion
        CUDA_CHECK(cudaMemcpyAsync(h_output + chunk * chunkSize, d_buf[currentBuf],
                                   thisChunkSize * sizeof(float),
                                   cudaMemcpyDeviceToHost, streamKernel));

        // Swap buffer index for next iteration
        currentBuf = 1 - currentBuf;
    }

    // Ensure all operations are complete
    CUDA_CHECK(cudaStreamSynchronize(streamCopy));
    CUDA_CHECK(cudaStreamSynchronize(streamKernel));

    // Verify results (simple check)
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_output[i] != h_input[i] + 1.0f) {
            fprintf(stderr, "Verification failed at index %d: got %f, expected %f\n",
                    i, h_output[i], h_input[i] + 1.0f);
            ok = false;
            break;
        }
    }
    if (ok) {
        printf("Verification succeeded.\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_buf[0]));
    CUDA_CHECK(cudaFree(d_buf[1]));
    CUDA_CHECK(cudaFreeHost(h_input));
    CUDA_CHECK(cudaFreeHost(h_output));
    CUDA_CHECK(cudaStreamDestroy(streamCopy));
    CUDA_CHECK(cudaStreamDestroy(streamKernel));

    return 0;
}
```