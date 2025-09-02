```cuda
/*
The main loop will be more complex. A common pattern is to prime the pipeline, then loop:
`for k=0 to N-1: stream = streams[k%2]; ... issue async work on stream... cudaStreamSynchronize(prev_stream); ... process result from prev_stream ...`

Thinking:
- Use two CUDA streams to overlap kernel execution with memory copies.
- Prime the pipeline by launching the first two iterations so that at each step
  we have one stream ready to work while the other finishes.
- In the main loop, synchronize on the *previous* stream before accessing its
  results, then launch the next iteration on the current stream.
- After the loop, make sure both streams are synchronized and process any
  remaining data.
- Use a simple element‑wise square kernel to illustrate the pattern.
- Provide error checking and clean up resources at the end.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

__global__ void square_kernel(const float *in, float *out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * in[idx];
}

int main()
{
    const size_t N      = 10;                 // number of pipeline iterations
    const size_t SIZE   = 1 << 20;            // array size per iteration
    const size_t BLOCKS = (SIZE + 255) / 256;
    const size_t THREADS = 256;

    // Allocate host memory
    float *h_in  = (float*)malloc(SIZE * sizeof(float));
    float *h_out = (float*)malloc(SIZE * sizeof(float));
    float *h_res = (float*)malloc(SIZE * sizeof(float));
    if (!h_in || !h_out || !h_res) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input data
    for (size_t i = 0; i < SIZE; ++i)
        h_in[i] = static_cast<float>(i) * 0.01f;

    // Allocate device memory
    float *d_in  = nullptr;
    float *d_out = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_in,  SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_out, SIZE * sizeof(float)));

    // Create two streams
    cudaStream_t streams[2];
    CHECK_CUDA(cudaStreamCreate(&streams[0]));
    CHECK_CUDA(cudaStreamCreate(&streams[1]]);

    // Prime the pipeline: launch first two iterations
    for (int k = 0; k < 2; ++k) {
        int stream = k % 2;
        CHECK_CUDA(cudaMemcpyAsync(d_in,  h_in, SIZE * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[stream]));
        square_kernel<<<BLOCKS, THREADS, 0, streams[stream]>>>(d_in, d_out, SIZE);
        CHECK_CUDA(cudaMemcpyAsync(h_out, d_out, SIZE * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[stream]));
    }

    // Main pipeline loop
    for (size_t k = 2; k < N; ++k) {
        int cur  = k % 2;
        int prev = (k - 1) % 2;

        // Wait for previous stream to finish before processing its result
        CHECK_CUDA(cudaStreamSynchronize(streams[prev]));
        // Example processing: accumulate sum of previous results
        double sum = 0.0;
        for (size_t i = 0; i < SIZE; ++i)
            sum += h_out[i];
        printf("Iteration %zu: sum of results = %f\n", k - 1, sum);

        // Launch next iteration on current stream
        CHECK_CUDA(cudaMemcpyAsync(d_in,  h_in, SIZE * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[cur]));
        square_kernel<<<BLOCKS, THREADS, 0, streams[cur]>>>(d_in, d_out, SIZE);
        CHECK_CUDA(cudaMemcpyAsync(h_out, d_out, SIZE * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[cur]));
    }

    // Final synchronization and processing
    CHECK_CUDA(cudaStreamSynchronize(streams[0]));
    CHECK_CUDA(cudaStreamSynchronize(streams[1]));
    // Process last results
    double final_sum = 0.0;
    for (size_t i = 0; i < SIZE; ++i)
        final_sum += h_out[i];
    printf("Final iteration: sum of results = %f\n", final_sum);

    // Clean up
    CHECK_CUDA(cudaStreamDestroy(streams[0]));
    CHECK_CUDA(cudaStreamDestroy(streams[1]));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    free(h_in);
    free(h_out);
    free(h_res);

    return 0;
}
```