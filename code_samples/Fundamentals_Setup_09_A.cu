/*
Add a comment to your program explaining what 'concurrent kernel execution' means and how it can improve performance.
This program demonstrates how to launch two simple kernels concurrently on separate CUDA streams.
It includes a comment that explains concurrent kernel execution: multiple kernels can execute
simultaneously on the GPU if the hardware supports it, allowing better utilization of compute
resources and overlapping of kernel execution with memory transfers or other kernels.
This can improve performance by reducing idle time and increasing throughput, especially
when the kernels are small or when the GPU has enough multiprocessors to run them in parallel.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that increments each element by 1
__global__ void inc_kernel(float *d_arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        d_arr[idx] += 1.0f;
}

// Simple kernel that multiplies each element by 2
__global__ void mul_kernel(float *d_arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        d_arr[idx] *= 2.0f;
}

// Helper macro for error checking
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

int main()
{
    const int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(float);

    // Host allocation
    float *h_arr = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) h_arr[i] = 1.0f;

    // Device allocation
    float *d_arr;
    CUDA_CHECK(cudaMalloc((void**)&d_arr, bytes));
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice));

    // Create two streams for concurrent execution
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // Launch inc_kernel on stream1
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    inc_kernel<<<grid, block, 0, stream1>>>(d_arr, N);

    // Launch mul_kernel on stream2
    mul_kernel<<<grid, block, 0, stream2>>>(d_arr, N);

    // Concurrent kernel execution means the GPU will try to run these kernels
    // at the same time if it has enough resources (multiprocessors, registers,
    // shared memory). This allows overlapping of kernel execution, potentially
    // reducing overall runtime compared to serial execution.

    // Synchronize streams to ensure kernels complete before we copy data back
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost));

    // Simple validation: after inc (+1) then mul (*2), each element should be (1+1)*2 = 4
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_arr[i] - 4.0f) > 1e-5f) {
            errors++;
            if (errors < 10) {
                printf("Mismatch at %d: %f\n", i, h_arr[i]);
            }
        }
    }
    printf("Validation %s. Errors: %d\n", errors == 0 ? "PASSED" : "FAILED", errors);

    // Clean up
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaFree(d_arr));
    free(h_arr);

    return 0;
}
