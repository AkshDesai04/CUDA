```cuda
/*********************************************************************
 * Aim of the program:
 * Use Nsight Systems (`nsys`) to visualize the timeline. You should
 * now see the HtoD copy for chunk `k+1` overlapping with the kernel
 * execution for chunk `k`.
 *
 * Thinking:
 * 1. We want a simple CUDA program that demonstrates overlapping
 *    host‑to‑device (HtoD) memory copies with kernel execution.
 * 2. The most common way to achieve this is to use double buffering
 *    with CUDA streams: while one stream is copying the next chunk
 *    of data to the device, the other stream runs the kernel on the
 *    previously copied chunk.  The host also asynchronously copies
 *    back the results, again using streams to avoid blocking.
 * 3. To make the effect visible in Nsight Systems, we need:
 *    - Pinned (page‑locked) host memory, so that asynchronous copies
 *      actually happen.
 *    - Multiple streams (at least two) to pipeline the operations.
 *    - A simple kernel that does not spend too much time; otherwise
 *      the copy might still be the bottleneck and the overlap may be
 *      masked.  We’ll use a trivial kernel that multiplies each
 *      element by a constant.
 * 4. The program will:
 *    - Allocate a large array on the host (pinned).
 *    - Allocate two device buffers (double buffer).
 *    - Create two CUDA streams.
 *    - Split the array into chunks; for each chunk, we:
 *        a. Asynchronously copy the chunk to the appropriate
 *           device buffer in stream `s`.
 *        b. Launch the kernel on that stream.
 *        c. Asynchronously copy the result back to the host in the
 *           same stream (so that the copy-back does not interfere
 *           with the next copy-to).
 *    - We will loop over the chunks so that at any time we have
 *      one chunk being processed and the next chunk being copied.
 *    - At the end we synchronize all streams.
 * 5. To see the overlap in Nsight Systems, we will compile with
 *    `-arch=sm_XX` (replace XX with your GPU compute capability)
 *    and run `nsys profile --stats=true --force-overwrite=true
 *    --output=timeline ./a.out`.  The timeline should show the
 *    HtoD copy for chunk k+1 overlapping the kernel for chunk k.
 *********************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHUNK_SIZE  (1 << 20)      // 1M elements per chunk
#define TOTAL_SIZE  (CHUNK_SIZE * 8) // 8 chunks
#define NUM_STREAMS 2

// Simple kernel: multiply each element by a constant factor
__global__ void scale_kernel(float *d_out, const float *d_in, float factor, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = d_in[idx] * factor;
    }
}

int main()
{
    // Check device
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Using device %d: %s\n", dev, prop.name);

    // Allocate pinned host memory
    float *h_in  = nullptr;
    float *h_out = nullptr;
    size_t bytes = TOTAL_SIZE * sizeof(float);
    cudaMallocHost(&h_in, bytes);
    cudaMallocHost(&h_out, bytes);

    // Initialize input data
    for (size_t i = 0; i < TOTAL_SIZE; ++i) {
        h_in[i] = static_cast<float>(i) * 0.001f;
    }

    // Allocate double buffered device memory
    float *d_buf[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaMalloc(&d_buf[i], CHUNK_SIZE * sizeof(float));
    }

    // Create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (CHUNK_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    float factor = 2.0f;

    size_t num_chunks = TOTAL_SIZE / CHUNK_SIZE;

    // Pipelined execution over chunks
    for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
        int streamIdx = chunk % NUM_STREAMS;

        size_t offset = chunk * CHUNK_SIZE;
        size_t size = CHUNK_SIZE * sizeof(float);

        // 1. Asynchronously copy HtoD
        cudaMemcpyAsync(d_buf[streamIdx], h_in + offset, size,
                        cudaMemcpyHostToDevice, streams[streamIdx]);

        // 2. Launch kernel
        scale_kernel<<<blocksPerGrid, threadsPerBlock, 0, streams[streamIdx]>>>(
            d_buf[streamIdx], d_buf[streamIdx], factor, CHUNK_SIZE);

        // 3. Asynchronously copy DtoH
        cudaMemcpyAsync(h_out + offset, d_buf[streamIdx], size,
                        cudaMemcpyDeviceToHost, streams[streamIdx]);
    }

    // Wait for all streams to finish
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    // Simple check: first few results
    printf("First 5 output values:\n");
    for (int i = 0; i < 5; ++i) {
        printf("%f ", h_out[i]);
    }
    printf("\n");

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaFree(d_buf[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);

    return 0;
}
```