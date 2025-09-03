```cuda
/*
Aim: 
The theoretical maximum speedup depends on the relative times of Copy, Exec, and Copy. 
If T_copy_HtoD + T_copy_DtoH > T_exec, the program is memory-bound and overlap is very effective.

Thinking:
To illustrate this principle, we create a small CUDA program that:
1. Allocates an array on the host and device.
2. Performs a simple element-wise computation on the device (kernel).
3. Measures the time for host-to-device copy, kernel execution, and device-to-host copy
   using CUDA events (blocking, serial execution).
4. Computes the serial total time and the ideal overlapped time.
   - The overlapped time is approximated as:
     T_overlap = T_copy_HtoD + max(T_exec, T_copy_DtoH)
     because the kernel cannot start until the HtoD copy finishes,
     but the DtoH copy can overlap with the kernel.
5. Calculates the theoretical maximum speedup as:
     speedup = serial_total_time / T_overlap
6. Determines whether the program is memory-bound by checking if
   T_copy_HtoD + T_copy_DtoH > T_exec.
7. Prints out the measured times, the overlapped time, the speedup,
   and whether the program is memory-bound.

The program uses a single kernel launch and simple timing; it does not
attempt to actually pipeline multiple chunks, but it demonstrates how
to compute the theoretical speedup based on the relative timings of
the copy and execution phases.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1 << 24) // 16M elements (~64MB for float)
#define BLOCK_SIZE 256

// Simple kernel that performs element-wise addition
__global__ void addKernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",     \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),          \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

int main(void) {
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    // Create CUDA events for timing
    cudaEvent_t h2d_start, h2d_end;
    cudaEvent_t kernel_start, kernel_end;
    cudaEvent_t d2h_start, d2h_end;

    CUDA_CHECK(cudaEventCreate(&h2d_start));
    CUDA_CHECK(cudaEventCreate(&h2d_end));
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_end));
    CUDA_CHECK(cudaEventCreate(&d2h_start));
    CUDA_CHECK(cudaEventCreate(&d2h_end));

    // --------- Serial (blocking) execution ---------
    // Host to Device copy
    CUDA_CHECK(cudaEventRecord(h2d_start));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(h2d_end));
    CUDA_CHECK(cudaEventSynchronize(h2d_end));

    // Kernel launch
    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + threads.x - 1) / threads.x);
    CUDA_CHECK(cudaEventRecord(kernel_start));
    addKernel<<<blocks, threads>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(kernel_end));
    CUDA_CHECK(cudaEventSynchronize(kernel_end));

    // Device to Host copy
    CUDA_CHECK(cudaEventRecord(d2h_start));
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(d2h_end));
    CUDA_CHECK(cudaEventSynchronize(d2h_end));

    // Retrieve elapsed times
    float t_h2d, t_kernel, t_d2h;
    CUDA_CHECK(cudaEventElapsedTime(&t_h2d, h2d_start, h2d_end));
    CUDA_CHECK(cudaEventElapsedTime(&t_kernel, kernel_start, kernel_end));
    CUDA_CHECK(cudaEventElapsedTime(&t_d2h, d2h_start, d2h_end));

    float serial_total = t_h2d + t_kernel + t_d2h;

    // --------- Overlapped execution (idealized) ---------
    // Note: In this simple example we only overlap the DtoH copy with the kernel,
    // because the kernel depends on the completion of the HtoD copy.
    // For a more realistic pipeline, data would be split into chunks.
    // We simulate overlapped time as:
    //   T_overlap = T_h2d + max(T_kernel, T_d2h)
    float overlapped_time = t_h2d + ((t_kernel > t_d2h) ? t_kernel : t_d2h);

    // Calculate theoretical maximum speedup
    float speedup = serial_total / overlapped_time;

    // Determine if memory-bound
    bool memory_bound = (t_h2d + t_d2h > t_kernel);

    // Print results
    printf("=== Timing Results (ms) ===\n");
    printf("Host to Device copy time: %8.3f ms\n", t_h2d);
    printf("Kernel execution time:    %8.3f ms\n", t_kernel);
    printf("Device to Host copy time: %8.3f ms\n", t_d2h);
    printf("Serial total time:        %8.3f ms\n", serial_total);
    printf("Ideal overlapped time:    %8.3f ms\n", overlapped_time);
    printf("Theoretical max speedup:  %8.3f\n", speedup);
    printf("Memory-bound condition:   %s\n",
           memory_bound ? "YES (Overlap is very effective)" : "NO");

    // Clean up
    CUDA_CHECK(cudaEventDestroy(h2d_start));
    CUDA_CHECK(cudaEventDestroy(h2d_end));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_end));
    CUDA_CHECK(cudaEventDestroy(d2h_start));
    CUDA_CHECK(cudaEventDestroy(d2h_end));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```