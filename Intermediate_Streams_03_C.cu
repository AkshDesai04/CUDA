/*
Aim: Try changing the 'work' done by the kernel. If the kernel is very fast, the speedup will be high. If the kernel is very slow (compute-bound), the speedup will be lower because the copies finish long before the kernel does.

Thinking:
The goal of this program is to illustrate how the amount of compute work performed by a CUDA kernel affects the overall speedup relative to memory copy operations. We allocate a large array on the host, copy it to the device, run a kernel that performs a user‑specified amount of compute work per element, and then copy the results back. By timing each stage (copy to device, kernel execution, copy back), we can observe how increasing the work factor changes the balance between compute and memory bandwidth. The program uses CUDA events for precise timing and a single stream for simplicity. The user can change the work factor via a command‑line argument; if omitted, a default moderate value is used.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define N (1 << 24)            // Size of the array (16M elements)
#define BLOCK_SIZE 256         // Threads per block

// Kernel that performs 'work_factor' amount of floating point operations per element
__global__ void compute_kernel(float *d_data, int work_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = d_data[idx];
        for (int i = 0; i < work_factor; ++i) {
            val = sinf(val) * cosf(val) + expf(val);
        }
        d_data[idx] = val;
    }
}

// Helper to check CUDA errors
void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    int work_factor = 10;  // Default work factor
    if (argc > 1) {
        work_factor = atoi(argv[1]);
        if (work_factor <= 0) work_factor = 10;
    }

    printf("Array size: %d elements\n", N);
    printf("Work factor per element: %d\n", work_factor);

    // Allocate host memory
    float *h_data = (float *)malloc(N * sizeof(float));
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    // Initialize data
    for (int i = 0; i < N; ++i) h_data[i] = static_cast<float>(i) * 0.001f;

    // Allocate device memory
    float *d_data = nullptr;
    checkCuda(cudaMalloc((void **)&d_data, N * sizeof(float)), "cudaMalloc");

    // Create a stream
    cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    // Create events for timing
    cudaEvent_t start, stop, copy_start, copy_stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    checkCuda(cudaEventCreate(&copy_start), "cudaEventCreate copy_start");
    checkCuda(cudaEventCreate(&copy_stop), "cudaEventCreate copy_stop");

    // Copy from host to device
    checkCuda(cudaEventRecord(copy_start, stream), "cudaEventRecord copy_start");
    checkCuda(cudaMemcpyAsync(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync H2D");
    checkCuda(cudaEventRecord(copy_stop, stream), "cudaEventRecord copy_stop");
    checkCuda(cudaEventSynchronize(copy_stop), "cudaEventSynchronize copy_stop");

    // Launch kernel
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    checkCuda(cudaEventRecord(start, stream), "cudaEventRecord start");
    compute_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(d_data, work_factor);
    checkCuda(cudaEventRecord(stop, stream), "cudaEventRecord stop");
    checkCuda(cudaGetLastError(), "Kernel launch");

    // Copy back from device to host
    checkCuda(cudaEventRecord(copy_start, stream), "cudaEventRecord copy_start after kernel");
    checkCuda(cudaMemcpyAsync(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync D2H");
    checkCuda(cudaEventRecord(copy_stop, stream), "cudaEventRecord copy_stop after kernel");
    checkCuda(cudaEventSynchronize(copy_stop), "cudaEventSynchronize copy_stop after kernel");

    // Compute elapsed times
    float time_copy, time_kernel, time_total;
    checkCuda(cudaEventElapsedTime(&time_copy, copy_start, copy_stop), "cudaEventElapsedTime copy");
    checkCuda(cudaEventElapsedTime(&time_kernel, start, stop), "cudaEventElapsedTime kernel");
    time_total = time_copy + time_kernel;

    printf("Time to copy H2D + D2H: %.3f ms\n", time_copy);
    printf("Kernel execution time: %.3f ms\n", time_kernel);
    printf("Total time: %.3f ms\n", time_total);

    // Optional: verify a few results
    bool ok = true;
    for (int i = 0; i < 10; ++i) {
        printf("h_data[%d] = %f\n", i, h_data[i]);
    }

    // Clean up
    checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    checkCuda(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
    checkCuda(cudaEventDestroy(copy_start), "cudaEventDestroy copy_start");
    checkCuda(cudaEventDestroy(copy_stop), "cudaEventDestroy copy_stop");
    checkCuda(cudaFree(d_data), "cudaFree");
    free(h_data);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
